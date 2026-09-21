from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        # Training first loads the plain pretrained model strictly, then wraps
        # Q/V. Evaluation reconstructs the wrapped checkpoint architecture here.
        if (self.args.training_stage == 'finetune' and
                self.args.finetune_strategy == 'lora' and not self.args.is_training):
            model.configure_lora(self.args.lora_rank, self.args.lora_alpha,
                                 self.args.lora_dropout)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        gradual = (self.args.training_stage == 'finetune' and
                   self.args.finetune_strategy == 'gradual')
        if gradual:
            # Register every eventual online parameter once. Frozen parameters
            # have grad=None, so Adam will skip them until their phase starts.
            self._core_model().configure_finetuning('full')
        trainable = [(name, parameter) for name, parameter in
                     self.model.named_parameters() if parameter.requires_grad]
        if gradual:
            self._configure_gradual_epoch(0)
        if not trainable:
            raise ValueError('No trainable parameters were selected')

        if self.args.training_stage != 'finetune':
            return optim.Adam(
                [parameter for _, parameter in trainable],
                lr=self.args.learning_rate)

        head_names = ('projector.', 'module.projector.',
                      'incident_output.', 'module.incident_output.')
        head_parameters = [parameter for name, parameter in trainable
                           if name.startswith(head_names)]
        backbone_parameters = [parameter for name, parameter in trainable
                               if not name.startswith(head_names)]
        parameter_groups = []
        backbone_scale = (self.args.lora_lr_scale
                          if self.args.finetune_strategy == 'lora'
                          else self.args.encoder_lr_scale)
        if backbone_parameters:
            parameter_groups.append({
                'params': backbone_parameters,
                'lr': self.args.learning_rate * backbone_scale,
                'lr_scale': backbone_scale,
            })
        if head_parameters:
            parameter_groups.append({
                'params': head_parameters,
                'lr': self.args.learning_rate,
                'lr_scale': 1.0,
            })
        return optim.Adam(parameter_groups)

    def _core_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _load_pretrained_for_finetuning(self):
        checkpoint = torch.load(
            self.args.pretrained_checkpoint, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        checkpoint = {
            (name[7:] if name.startswith('module.') else name): value
            for name, value in checkpoint.items()
        }
        core_model = self._core_model()
        # Preserve the online encoder, predictor and trained forecasting head
        # together. Replacing the encoder with its EMA copy changes this mapping.
        core_model.load_state_dict(checkpoint, strict=True)
        if self.args.finetune_strategy == 'lora':
            core_model.configure_lora(self.args.lora_rank, self.args.lora_alpha,
                                      self.args.lora_dropout)
        else:
            core_model.configure_finetuning(
                'partial' if self.args.finetune_strategy == 'gradual' else self.args.finetune_strategy,
                0 if self.args.finetune_strategy == 'gradual' else self.args.partial_unfreeze_layers)
        trainable = sum(parameter.numel() for parameter in core_model.parameters()
                        if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in core_model.parameters())
        print(f'Loaded JEPA checkpoint: {self.args.pretrained_checkpoint}')
        print(f'Fine-tune strategy: {self.args.finetune_strategy}; '
              f'trainable parameters: {trainable:,}/{total:,} '
              f'({100.0 * trainable / total:.2f}%)')

    def _configure_gradual_epoch(self, epoch):
        """Zero-based epoch; preserve optimizer moments across unfreezing."""
        if epoch < self.args.gradual_head_epochs:
            phase, strategy, layers = 'predictor/head', 'partial', 0
        elif epoch < self.args.gradual_head_epochs + self.args.gradual_partial_epochs:
            phase, strategy, layers = 'last encoder layers', 'partial', self.args.partial_unfreeze_layers
        else:
            phase, strategy, layers = 'full online network', 'full', 0
        model = self._core_model()
        model.configure_finetuning(strategy, layers)
        # Graph/incident feature extractors join only the final full phase.
        if strategy != 'full':
            for name in ('incident_fusion', 'graph_encoder', 'fusion_gate'):
                if hasattr(model, name):
                    getattr(model, name).requires_grad_(False)
        if getattr(self, '_gradual_phase', None) != phase:
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'Gradual epoch {epoch + 1}: {phase}; trainable={count:,}')
            self._gradual_phase = phase

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _forecast_pretrain_loss(self, forecast, future, mask_info):
        """Weighted MSE plus anti-degeneracy regularizers for learned masks."""
        unweighted = nn.functional.mse_loss(forecast, future)
        zero = unweighted.new_zeros(())
        if mask_info is None:
            return unweighted, {
                'forecast': unweighted, 'forecast_weighted': unweighted,
                'mask_budget': zero, 'mask_neg_entropy': zero,
                'mask_smooth': zero, 'mask_mean': unweighted.new_ones(()),
                'mask_min': unweighted.new_ones(()),
                'mask_max': unweighted.new_ones(()),
            }

        weights = mask_info['weights']
        probabilities = mask_info['probabilities'].clamp(1e-6, 1.0 - 1e-6)
        if weights.shape != forecast.shape:
            raise ValueError(
                f'Forecast mask {weights.shape} does not match forecast {forecast.shape}')
        squared_error = (forecast - future).square()
        weighted = (weights * squared_error).sum() / weights.sum().clamp_min(1e-6)
        budget = (weights.mean() - self.args.forecast_mask_target).square()
        negative_entropy = (probabilities * probabilities.log() +
                            (1.0 - probabilities) *
                            (1.0 - probabilities).log()).mean()
        smooth = (weights[:, 1:] - weights[:, :-1]).abs().mean()
        total = (weighted + self.args.forecast_mask_budget_weight * budget +
                 self.args.forecast_mask_entropy_weight * negative_entropy +
                 self.args.forecast_mask_smooth_weight * smooth)
        return total, {
            'forecast': unweighted, 'forecast_weighted': weighted,
            'mask_budget': budget, 'mask_neg_entropy': negative_entropy,
            'mask_smooth': smooth, 'mask_mean': weights.mean(),
            'mask_min': weights.min(), 'mask_max': weights.max(),
        }

    def _unpack_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
        incident_data = None
        target_text = None
        if getattr(self.args, 'incident', False):
            if len(batch) < 7:
                raise ValueError(
                    '--incident expects incident_features, incident_position, '
                    'and incident_distances after the four forecasting tensors')
            incident_data = {
                'features': batch[4].float().to(self.device),
                'position': batch[5].long().to(self.device),
                'distances': batch[6].float().to(self.device),
            }
        elif len(batch) > 4:
            target_text = batch[4].float().to(self.device)
        return (batch_x, batch_y, batch_x_mark, batch_y_mark,
                incident_data, target_text)

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark,
                       incident_data):
        if incident_data is not None:
            return self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                incident_data=incident_data)
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                (batch_x, batch_y, batch_x_mark, batch_y_mark,
                 incident_data, _) = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, self.args.label_len:self.args.label_len + self.args.pred_len, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)[0]
                        else:
                            outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)
                else:
                    if self.args.output_attention:
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)[0]
                    else:
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.pred_len, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _pretrain_epoch(self, data_loader, optimizer=None, scaler=None):
        is_training = optimizer is not None
        self.model.train(is_training)
        history = {
            'forecast': [], 'forecast_weighted': [],
            'mask_budget': [], 'mask_neg_entropy': [], 'mask_smooth': [],
            'mask_mean': [], 'mask_min': [], 'mask_max': [],
            'jepa': [], 'topo': [], 'text': [], 'alignment': [], 'total': []
        }
        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for batch in data_loader:
                (batch_x, batch_y, batch_x_mark, batch_y_mark,
                 incident_data, target_text) = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                target_x = batch_y[
                    :, self.args.label_len:
                    self.args.label_len + self.args.seq_len, :]
                target_mark = (batch_y_mark[
                    :, self.args.label_len:
                    self.args.label_len + self.args.seq_len, :]
                    if batch_y_mark is not None else None)
                dec_inp = torch.zeros_like(batch_y).to(self.device)

                if is_training:
                    optimizer.zero_grad()
                amp_context = (torch.cuda.amp.autocast()
                               if self.args.use_amp else
                               torch.cuda.amp.autocast(enabled=False))
                with amp_context:
                    (forecast, jepa_loss, topo_loss, text_loss,
                     alignment_loss, mask_info) = self._core_model().forward_with_jepa(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        target_x, target_mark, target_text,
                        incident_data=incident_data,
                        return_forecast_mask=True)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    future = batch_y[:, self.args.label_len:
                                     self.args.label_len + self.args.pred_len, f_dim:]
                    forecast_objective, mask_metrics = self._forecast_pretrain_loss(
                        forecast[:, -self.args.pred_len:, f_dim:], future,
                        mask_info)
                    loss = (forecast_objective + self.args.jepa_weight * jepa_loss +
                            self.args.topo_weight * topo_loss +
                            self.args.text_weight * text_loss +
                            self.args.alignment_weight * alignment_loss)

                if is_training:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    self._core_model().update_target_encoder()

                values = {
                    **{name: value.item() for name, value in mask_metrics.items()},
                    'jepa': jepa_loss.item(),
                    'topo': topo_loss.item(),
                    'text': text_loss.item(),
                    'alignment': alignment_loss.item(),
                    'total': loss.item(),
                }
                for name, value in values.items():
                    history[name].append(value)

        return {name: float(np.average(values))
                for name, values in history.items()}

    def pretrain(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        core_model = self._core_model()
        core_model.configure_pretraining()
        trainable = sum(parameter.numel() for parameter in core_model.parameters()
                        if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in core_model.parameters())
        print(f'JEPA pretraining trainable parameters: {trainable:,}/{total:,} '
              f'({100.0 * trainable / total:.2f}%)')

        optimizer = self._select_optimizer()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_metrics = self._pretrain_epoch(
                train_loader, optimizer=optimizer, scaler=scaler)
            val_metrics = self._pretrain_epoch(vali_loader)
            print(
                'Pretrain Epoch: {0} | train total: {1:.7f}, '
                'jepa: {2:.7f} | val total: {3:.7f}, jepa: {4:.7f} '
                '| time: {5:.2f}s'.format(
                    epoch + 1, train_metrics['total'],
                    train_metrics['jepa'], val_metrics['total'],
                    val_metrics['jepa'], time.time() - epoch_time))
            print('Forecast MSE | train: {:.7f} | val: {:.7f}'.format(
                train_metrics['forecast'], val_metrics['forecast']))
            if self.args.forecast_mask:
                print(
                    'Masked forecast MSE | train: {:.7f} | val: {:.7f} | '
                    'mask mean/min/max (val): {:.4f}/{:.4f}/{:.4f}'.format(
                        train_metrics['forecast_weighted'],
                        val_metrics['forecast_weighted'],
                        val_metrics['mask_mean'], val_metrics['mask_min'],
                        val_metrics['mask_max']))
            # All stages select checkpoints using the same forecasting metric.
            early_stopping(val_metrics['forecast'], self.model, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(
            best_model_path, map_location=self.device))
        print(f'Best pretraining checkpoint: {best_model_path}')
        return self.model

    def test_pretrain(self, setting, load=False):
        if load:
            checkpoint_path = os.path.join(
                self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(
                checkpoint_path, map_location=self.device))
        # The head was trained during pretraining: evaluate traffic directly.
        return self.test(setting)

    def train(self, setting):
        if self.args.training_stage == 'pretrain':
            return self.pretrain(setting)
        if self.args.training_stage == 'finetune':
            self._load_pretrained_for_finetuning()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        if getattr(self.args, 'fremont_adaptation_root', ''):
            test_data, test_loader = None, None
        else:
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            if (self.args.training_stage == 'finetune' and
                    self.args.finetune_strategy == 'gradual'):
                self._configure_gradual_epoch(epoch)
            train_loss = []
            branch_history = {
                'forecast': [], 'jepa': [], 'topo': [],
                'text': [], 'alignment': [], 'total': []
            }

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                (batch_x, batch_y, batch_x_mark, batch_y_mark,
                 incident_data, target_text) = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                zero_loss = batch_x.new_zeros(())
                jepa_loss = zero_loss
                topo_loss = zero_loss
                text_loss = zero_loss
                alignment_loss = zero_loss

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, self.args.label_len:self.args.label_len + self.args.pred_len, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                use_jepa = (self.args.training_stage == 'joint' and
                            self.args.model_variant == 'jepa' and
                            (self.args.jepa_weight > 0 or self.args.topo_weight > 0 or
                             self.args.text_weight > 0 or
                             self.args.alignment_weight > 0))
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if use_jepa:
                            core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                            target_x = batch_y[:, self.args.label_len:self.args.label_len + self.args.seq_len, :]
                            target_mark = batch_y_mark[:, self.args.label_len:self.args.label_len + self.args.seq_len, :] if batch_y_mark is not None else None
                            outputs, jepa_loss, topo_loss, text_loss, alignment_loss = core_model.forward_with_jepa(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                target_x, target_mark, target_text,
                                incident_data=incident_data)
                        elif self.args.output_attention:
                            outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)[0]
                        else:
                            outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.pred_len, f_dim:].to(self.device)
                        forecast_loss = criterion(outputs, batch_y)
                        loss = (forecast_loss + self.args.jepa_weight * jepa_loss +
                                self.args.topo_weight * topo_loss +
                                self.args.text_weight * text_loss +
                                self.args.alignment_weight * alignment_loss) if use_jepa else forecast_loss
                else:
                    if use_jepa:
                        core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                        target_x = batch_y[:, self.args.label_len:self.args.label_len + self.args.seq_len, :]
                        target_mark = batch_y_mark[:, self.args.label_len:self.args.label_len + self.args.seq_len, :] if batch_y_mark is not None else None
                        outputs, jepa_loss, topo_loss, text_loss, alignment_loss = core_model.forward_with_jepa(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            target_x, target_mark, target_text,
                            incident_data=incident_data)
                    elif self.args.output_attention:
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)[0]
                    else:
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.pred_len, f_dim:].to(self.device)
                    forecast_loss = criterion(outputs, batch_y)
                    loss = (forecast_loss + self.args.jepa_weight * jepa_loss +
                            self.args.topo_weight * topo_loss +
                            self.args.text_weight * text_loss +
                            self.args.alignment_weight * alignment_loss) if use_jepa else forecast_loss
                branch_values = {
                    'forecast': forecast_loss.item(),
                    'jepa': jepa_loss.item(),
                    'topo': topo_loss.item(),
                    'text': text_loss.item(),
                    'alignment': alignment_loss.item(),
                    'total': loss.item(),
                }
                train_loss.append(branch_values['total'])
                for branch_name, branch_value in branch_values.items():
                    branch_history[branch_name].append(branch_value)

                if (i + 1) % self.args.log_interval == 0:
                    print(
                        '\titers: {0}, epoch: {1} | total: {2:.7f} | '
                        'forecast: {3:.7f} | jepa: {4:.7f} (weighted: {5:.7f}) | '
                        'topo: {6:.7f} (weighted: {7:.7f}) | '
                        'text: {8:.7f} (weighted: {9:.7f}) | '
                        'alignment: {10:.7f} (weighted: {11:.7f})'.format(
                            i + 1, epoch + 1, branch_values['total'],
                            branch_values['forecast'], branch_values['jepa'],
                            self.args.jepa_weight * branch_values['jepa'],
                            branch_values['topo'],
                            self.args.topo_weight * branch_values['topo'],
                            branch_values['text'],
                            self.args.text_weight * branch_values['text'],
                            branch_values['alignment'],
                            self.args.alignment_weight * branch_values['alignment']))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if use_jepa:
                    core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    core_model.update_target_encoder()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            branch_means = {
                name: float(np.average(values))
                for name, values in branch_history.items()
            }
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = (self.vali(test_data, test_loader, criterion)
                         if test_loader is not None else float('nan'))

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if (self.args.training_stage == 'finetune' and
                    self.args.finetune_strategy == 'gradual' and
                    epoch < self.args.gradual_head_epochs + self.args.gradual_partial_epochs):
                # Keep the global best checkpoint but allow all phases to run.
                early_stopping.counter = 0
                early_stopping.early_stop = False
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(
                self.args.checkpoints, setting, 'checkpoint.pth'),
                map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                (batch_x, batch_y, batch_x_mark, batch_y_mark,
                 incident_data, _) = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, self.args.label_len:self.args.label_len + self.args.pred_len, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)[0]
                        else:
                            outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)
                else:
                    if self.args.output_attention:
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)[0]

                    else:
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, incident_data)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.pred_len, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, self.args.label_len:self.args.label_len + self.args.pred_len, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
