"""Paired-window pretraining and isolated traffic transfer experiments."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import TrafficWindows
from model import Config, SCJEPA
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Preprocess'))
from split_fremont_adaptation import verify_split


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(path, device):
    # Generated checkpoints contain tensors and plain Python metadata only.
    return torch.load(path, map_location=device, weights_only=True)


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def loader(dataset, args, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, drop_last=False,
                      generator=torch.Generator().manual_seed(args.seed))


@torch.no_grad()
def score(model, batches, device, objective=False, forecast_weight=0.):
    model.eval()
    absolute = squared = count = 0
    sums, samples = {}, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        if objective:
            loss, terms = model.objective(x, y, forecast_weight=forecast_weight)
            terms = dict(terms, total=loss)
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.) + value.item() * len(x)
            samples += len(x)
        else:
            error = (model(x) - y).double()
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
    result = ({key: value / samples for key, value in sums.items()} if objective else
              {'MAE': absolute / count, 'MSE': squared / count, 'RMSE': math.sqrt(squared / count)})
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite evaluation result')
    return result


def evaluate_checkpoint(path, batches, device, output_dir=None):
    checkpoint = load_checkpoint(path, device)
    if checkpoint.get('config', {}).get('architecture') != 'scjepa_fine_only_v1':
        raise ValueError('Legacy/incompatible SC-JEPA checkpoint; pretrain this architecture again')
    model = SCJEPA(Config(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    metrics = score(model, batches, device)
    write_json((output_dir or path.parent) / 'metrics.json', metrics)
    print(f'{path.parent.name}: {metrics}', flush=True)
    return metrics


def fit(stage, args, config, datasets, device, source=None):
    seed_all(args.seed)
    destination = Path(args.run_dir) / stage
    # A new run directory or removing/archiving the old run is an explicit user decision.
    destination.mkdir(parents=True, exist_ok=False)
    model = SCJEPA(config).to(device)
    source_hash = None
    if source is not None:
        checkpoint = load_checkpoint(source, device)
        if checkpoint['config'] != asdict(config):
            raise ValueError('Pretraining architecture differs; pass matching model arguments')
        if checkpoint['stage'] != 'pretrain' or checkpoint['data_contract'] != data_contract(args):
            raise ValueError('Pretraining data contract differs from this run')
        model.load_state_dict(checkpoint['model'], strict=True)
        source_hash = digest(source)
    model.configure(stage, args.forecast_weight)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'{stage}: trainable={trainable:,}/{total:,}; source={source}', flush=True)
    groups = []
    for name, parameters, scale in (
            ('head', list(model.forecast_head.parameters()), 1.),
            ('backbone', [p for n, p in model.named_parameters()
                          if not n.startswith('forecast_head.')],
             1. if stage == 'pretrain' else args.encoder_lr_scale)):
        parameters = [p for p in parameters if p.requires_grad]
        if parameters:
            groups.append(dict(params=parameters, lr=args.lr * scale, name=name))
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_gamma)
    train_batches = loader(datasets['train'], args, True)
    val_batches = loader(datasets['val'], args)
    epochs = args.pretrain_epochs if stage == 'pretrain' else args.finetune_epochs
    best, stale, history = float('inf'), 0, []
    start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        model.train()
        train_sum, seen, terms_sum = 0., 0, {}
        epoch_start = time.perf_counter()
        for x, y in train_batches:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if stage == 'pretrain':
                loss, terms = model.objective(x, y, args.forecast_weight)
                for key, value in terms.items():
                    terms_sum[key] = terms_sum.get(key, 0.) + value.item() * len(x)
            else:
                loss = (model(x) - y).square().mean()
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at {stage} epoch {epoch + 1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
            optimizer.step()
            if stage == 'pretrain':
                model.update_ema()
            train_sum += loss.item() * len(x)
            seen += len(x)
        validation = score(model, val_batches, device, stage == 'pretrain', args.forecast_weight)
        # Pure SSL selects by the retained fine-scale KL validation objective.
        # Joint pretraining selects by forecast MSE, matching the existing JEPA protocol.
        key = ('total' if args.forecast_weight == 0 else 'forecast_mse') if stage == 'pretrain' else 'MSE'
        record = dict(epoch=epoch + 1, train_loss=train_sum / seen, validation=validation,
                      lr=[group['lr'] for group in optimizer.param_groups],
                      seconds=time.perf_counter() - epoch_start,
                      train_terms={k: v / seen for k, v in terms_sum.items()})
        history.append(record)
        print(f'{stage} epoch {epoch + 1}/{epochs}: train={train_sum / seen:.6f} '
              f'val_{key}={validation[key]:.6f}', flush=True)
        if validation[key] < best:
            best, stale = validation[key], 0
            torch.save(dict(model=model.state_dict(), config=asdict(config), stage=stage,
                            epoch=epoch + 1, validation=validation,
                            forecast_weight=args.forecast_weight, source_sha256=source_hash,
                            data_contract=data_contract(args, stage),
                            evaluation_contract=data_contract(args, 'test')),
                       destination / 'checkpoint.pt')
        else:
            stale += 1
        write_json(destination / 'history.json', history)
        scheduler.step()
        if stale >= args.patience:
            break
    write_json(destination / 'run.json', dict(
        arguments=vars(args), model=asdict(config), source_sha256=source_hash,
        trainable_parameters=trainable, total_parameters=total,
        seconds=time.perf_counter() - start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        split_sizes={k: len(v) for k, v in datasets.items()}))
    path = destination / 'checkpoint.pt'
    if args.adaptation_dir:
        (destination / 'split_manifest.json').write_text(
            (Path(args.adaptation_dir) / 'split_manifest.json').read_text())
    # The test split is used only once, after validation checkpoint selection.
    if stage != 'pretrain' or args.forecast_weight > 0:
        evaluate_checkpoint(path, loader(datasets['test'], args), device)
    return path


def data_contract(args, stage='pretrain'):
    root = args.adaptation_dir if args.adaptation_dir and stage != 'pretrain' else args.data_dir
    return dict(root=str(Path(root).resolve()), feature=args.feature,
                pattern=args.pattern, nodes=args.nodes, seq_len=args.seq_len, pred_len=args.pred_len)


def summary(run_dir, adaptation=False):
    print('\nSC-JEPA traffic evaluation (stored traffic units)')
    print(f'{"Method":36s} {"MAE":>12s} {"MSE":>12s} {"RMSE":>12s}')
    for stage, label in [('baseline' if adaptation and (Path(run_dir) / 'baseline').exists() else 'pretrain', 'Joint pretrain direct test'),
                         ('probe', 'Fine-KL pretrain + frozen probe'),
                         ('full', 'Fine-KL pretrain + full fine-tune')]:
        path = Path(run_dir) / stage / 'metrics.json'
        if path.exists():
            m = json.loads(path.read_text())
            print(f'{label:36s} {m["MAE"]:12.7f} {m["MSE"]:12.7f} {m["RMSE"]:12.7f}')
        else:
            print(f'{label:36s} unavailable (SSL has no direct forecast evaluation)' if stage == 'pretrain'
                  else f'{label:36s} not run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all', 'transfer', 'pretrain', 'probe', 'full', 'eval', 'summary'], default='all')
    parser.add_argument('--data-dir', default='../TopoJEPA/dataset/Fremont')
    parser.add_argument('--adaptation-dir', default='', help='Verified former-test 60/20/20 split')
    parser.add_argument('--pattern', default='incident_{flag}.npy')
    parser.add_argument('--feature', type=int, default=0)
    parser.add_argument('--nodes', type=int, default=93)
    parser.add_argument('--run-dir', default='runs/fremont_fine_only_seed2026')
    parser.add_argument('--checkpoint', help='Source pretrain checkpoint, or checkpoint to evaluate')
    parser.add_argument('--architecture', choices=['scjepa_fine_only_v1'],
                        default='scjepa_fine_only_v1')
    for name, default in [('seq-len', 12), ('pred-len', 12), ('patch-len', 3), ('dim', 256),
                          ('codes', 128), ('heads', 8), ('layers', 6),
                          ('encoder-hidden', 256), ('predictor-dim', 128),
                          ('predictor-heads', 4), ('predictor-layers', 2), ('batch-size', 32),
                          ('pretrain-epochs', 20), ('finetune-epochs', 20), ('patience', 5),
                          ('workers', 0), ('seed', 2026)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in [('lr', .0001), ('encoder-lr-scale', .1), ('lr-gamma', .5),
                          ('dropout', .1), ('ema', .996), ('code-temperature', .1),
                          ('prediction-temperature', .8), ('forecast-weight', 0.)]:
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--device', default='auto', help='auto, cpu or cuda:0 etc.')
    args = parser.parse_args()
    if args.mode == 'summary':
        summary(args.run_dir, bool(args.adaptation_dir))
        return
    if args.adaptation_dir:
        verify_split(args.adaptation_dir, args.data_dir, args.pattern,
                     args.feature, args.seq_len, args.pred_len)
        if args.run_dir == 'runs/fremont_fine_only_seed2026':
            parser.error('Use a new --run-dir for adaptation experiments')
        run_root = Path(args.run_dir)
        manifest_bytes = (Path(args.adaptation_dir) / 'split_manifest.json').read_bytes()
        saved_manifest = run_root / 'adaptation_manifest.json'
        if saved_manifest.exists():
            if saved_manifest.read_bytes() != manifest_bytes:
                parser.error('Run directory belongs to another adaptation split')
        elif run_root.exists() and any(run_root.iterdir()):
            parser.error('Use a clean run directory, not an old-protocol result directory')
        else:
            run_root.mkdir(parents=True, exist_ok=True)
            saved_manifest.write_bytes(manifest_bytes)
    if (min(args.batch_size, args.pretrain_epochs, args.finetune_epochs, args.patience, args.nodes) <= 0 or
            args.workers < 0 or args.feature < 0 or args.forecast_weight < 0 or
            args.lr <= 0 or not 0 < args.encoder_lr_scale <= 1 or not 0 < args.lr_gamma <= 1 or
            any(not math.isfinite(v) for v in vars(args).values() if isinstance(v, float))):
        parser.error('Invalid training hyperparameters')
    config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if args.mode == 'eval':
        if not args.checkpoint:
            parser.error('--mode eval requires --checkpoint')
        ckpt = load_checkpoint(args.checkpoint, device)
        if ckpt['stage'] == 'pretrain' and ckpt['forecast_weight'] == 0:
            parser.error('SSL forecast head is untrained; run probe or full first')
        if ckpt.get('evaluation_contract', ckpt['data_contract']) != data_contract(args, 'test'):
            parser.error('Evaluation data contract differs from checkpoint')
        test = TrafficWindows(args.adaptation_dir or args.data_dir, 'test', args.seq_len, args.pred_len,
                              args.feature, args.nodes, args.pattern)
        evaluate_checkpoint(Path(args.checkpoint), loader(test, args), device)
        return
    datasets = {split: TrafficWindows(args.adaptation_dir if args.adaptation_dir and split == 'test'
                                     else args.data_dir, split, args.seq_len, args.pred_len,
                                     args.feature, args.nodes, args.pattern)
                for split in ('train', 'val', 'test')}
    print(f'Device={device}; split sizes=' + str({k: len(v) for k, v in datasets.items()}), flush=True)
    source = Path(args.checkpoint) if args.checkpoint else Path(args.run_dir) / 'pretrain/checkpoint.pt'
    if args.mode in ('all', 'pretrain'):
        source = fit('pretrain', args, config, datasets, device)
    if args.adaptation_dir:
        datasets = {split: TrafficWindows(args.adaptation_dir, split, args.seq_len, args.pred_len,
                                         args.feature, args.nodes, args.pattern)
                    for split in ('train', 'val', 'test')}
    if args.mode == 'transfer':
        if not args.adaptation_dir:
            parser.error('transfer requires --adaptation-dir')
        base = load_checkpoint(source, device)
        if base.get('config', {}).get('architecture') != config.architecture:
            raise ValueError('Legacy/incompatible SC-JEPA checkpoint; pretrain this architecture again')
        if base['stage'] != 'pretrain' or base['data_contract'] != data_contract(args):
            raise ValueError('Invalid source pretrain checkpoint/data contract')
        if base['forecast_weight'] > 0:
            baseline_dir = Path(args.run_dir) / 'baseline'
            baseline_dir.mkdir(parents=True, exist_ok=False)
            evaluate_checkpoint(source, loader(datasets['test'], args), device, baseline_dir)
            write_json(baseline_dir / 'evaluation.json', dict(
                checkpoint=str(source.resolve()), sha256=digest(source),
                evaluation_contract=data_contract(args, 'test')))
        else:
            print('SSL checkpoint has no trained forecast head; direct baseline unavailable.')
    for stage in ('probe', 'full'):
        if args.mode in ('all', 'transfer', stage):
            fit(stage, args, config, datasets, device, source)
    summary(args.run_dir, bool(args.adaptation_dir))


if __name__ == '__main__':
    main()
