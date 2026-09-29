"""Split the former test pool once for both JEPA implementations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def traffic_signature(sample, feature=0, seq_len=12, pred_len=12):
    """Hash exactly the float32 history/target values consumed by the models."""
    x, y = np.asarray(sample['x_data']), np.asarray(sample['y_data'])
    if (x.ndim != 3 or y.ndim != 3 or x.shape[0] != seq_len or
            y.shape[0] < pred_len or x.shape[1] != y.shape[1] or
            not 0 <= feature < min(x.shape[2], y.shape[2])):
        raise ValueError(f'Invalid traffic shape/feature: x={x.shape}, y={y.shape}')
    h = hashlib.sha256()
    for a in (x[:, :, feature], y[:pred_len, :, feature]):
        a = np.array(a, dtype='<f4', order='C', copy=True)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite traffic values cannot be deduplicated')
        a[a == 0] = 0  # Canonicalize signed zero, without rounding other values.
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def verify_split(root, source_root=None, pattern='incident_{flag}.npy',
                 feature=None, seq_len=None, pred_len=None):
    root = Path(root)
    manifest = json.loads((root / 'split_manifest.json').read_text())
    if manifest['protocol'] not in ('former_test_60_20_20_v1', 'former_test_60_20_20_dedup_v2'):
        raise ValueError('Unsupported adaptation manifest')
    dedup = manifest.get('deduplication')
    if manifest['protocol'].endswith('dedup_v2'):
        if not dedup:
            raise ValueError('Missing deduplication metadata')
        for key, value in [('feature', feature), ('seq_len', seq_len), ('pred_len', pred_len)]:
            if value is not None and value != dedup[key]:
                raise ValueError(f'Deduplication {key} differs from model data settings')
    if source_root is not None:
        source = Path(source_root) / pattern.format(flag='test')
        if sha256(source) != manifest['source_sha256']:
            raise ValueError('Adaptation pool does not match the original test file')
    used = set()
    for split in ('train', 'val', 'test'):
        info = manifest['splits'][split]
        if info['filename'] != pattern.format(flag=split):
            raise ValueError('File pattern differs from manifest')
        if sha256(root / info['filename']) != info['sha256']:
            raise ValueError(f'Adaptation {split} file changed since splitting')
        indices = info['source_indices']
        if (not indices or used.intersection(indices) or len(set(indices)) != len(indices)
                or info['count'] != len(indices)):
            raise ValueError('Empty or overlapping adaptation split')
        used.update(indices)
    dropped = set()
    if dedup:
        for pair in dedup['removed_to_retained']:
            index, retained = pair['removed'], pair['retained']
            if index in dropped or index in used or retained not in used:
                raise ValueError('Invalid duplicate-to-retained index mapping')
            dropped.add(index)
        if (len(used) != manifest['retained_count'] or len(dropped) != dedup['removed_count']):
            raise ValueError('Deduplication counts disagree with split indices')
    if used | dropped != set(range(manifest['source_count'])):
        raise ValueError('Manifest does not partition the source pool')
    return manifest


def create_split(source, destination, time_key=None, group_key=None,
                 acknowledge_stored_order=False, feature=0, seq_len=12, pred_len=12):
    source, destination = Path(source), Path(destination)
    if not time_key and not acknowledge_stored_order:
        raise ValueError('Supply --time-key or explicitly acknowledge unverified stored order')
    if feature < 0 or min(seq_len, pred_len) <= 0:
        raise ValueError('feature must be nonnegative and lengths positive')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {destination}')
    samples = np.load(source, allow_pickle=True)  # trusted local pickle data only
    n = len(samples)
    if n < 5 or any(not isinstance(s, dict) or not {'x_data', 'y_data'} <= s.keys() for s in samples):
        raise ValueError('Need at least five x_data/y_data dictionary samples')
    indices = list(range(n))
    if time_key:
        # Caller supplies comparable numeric times or ISO timestamps, not time-of-day alone.
        indices.sort(key=lambda i: samples[i][time_key])
    source_count = n
    unique, seen, removed = [], {}, []
    for i in indices:
        signature = traffic_signature(samples[i], feature, seq_len, pred_len)
        if signature in seen:
            removed.append(dict(removed=i, retained=seen[signature]))
        else:
            seen[signature] = i
            unique.append(i)
    indices = unique
    n = len(indices)
    if n < 5:
        raise ValueError(f'Only {n} unique traffic windows remain; need at least five')
    bounds = [int(n * .6), int(n * .8)]
    if group_key:
        groups = [str(samples[i][group_key]) for i in indices]
        finished, previous = set(), None
        for group in groups:
            if group != previous:
                if group in finished:
                    raise ValueError('Noncontiguous event groups: provide appropriate chronological ordering')
                if previous is not None:
                    finished.add(previous)
                previous = group
        allowed = [i for i in range(1, n) if groups[i - 1] != groups[i]]
        if len(allowed) < 2:
            raise ValueError('Need at least three contiguous event groups')
        first = min(allowed[:-1], key=lambda i: abs(i - bounds[0]))
        second = min([i for i in allowed if i > first], key=lambda i: abs(i - bounds[1]))
        bounds = [first, second]
    parts = dict(zip(('train', 'val', 'test'),
                     (indices[:bounds[0]], indices[bounds[0]:bounds[1]], indices[bounds[1]:])))
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol='former_test_60_20_20_dedup_v2', source=str(source.resolve()),
                    source_sha256=sha256(source), source_count=source_count, retained_count=n,
                    deduplication=dict(feature=feature, seq_len=seq_len, pred_len=pred_len,
                        comparison='exact_float32_traffic_xy', keep='first_in_effective_order',
                        removed_count=len(removed), removed_to_retained=removed), ratios=[.6, .2, .2],
                    time_key=time_key, group_key=group_key,
                    order='sorted_by_supplied_time_key' if time_key else 'stored_order_unverified',
                    warning='Previously inspected test data. Partial window overlap and temporal independence not certified.',
                    splits={})
    for split, ids in parts.items():
        path = destination / f'incident_{split}.npy'
        np.save(path, samples[ids], allow_pickle=True)
        manifest['splits'][split] = dict(filename=path.name, count=len(ids),
                                        source_indices=ids, sha256=sha256(path))
    (destination / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--time-key', help='Scalar numeric time or ISO timestamp field')
    parser.add_argument('--group-key', help='Event ID field, keep entire contiguous groups together')
    parser.add_argument('--acknowledge-stored-order', action='store_true')
    parser.add_argument('--feature', type=int, default=0, help='Traffic channel used by the model')
    parser.add_argument('--seq-len', type=int, default=12)
    parser.add_argument('--pred-len', type=int, default=12)
    args = parser.parse_args()
    manifest = create_split(args.source, args.output, args.time_key, args.group_key,
                            args.acknowledge_stored_order, args.feature, args.seq_len, args.pred_len)
    print(f"Source: {manifest['source_count']}; unique traffic windows: {manifest['retained_count']}; "
          f"removed from derived splits: {manifest['deduplication']['removed_count']}")
    print({k: v['count'] for k, v in manifest['splits'].items()})
    print(manifest['order'], manifest['warning'])
