import os


def clip_cache_name(data_path, flag, seq_len, pred_len, future_len):
    stem = os.path.splitext(os.path.basename(data_path))[0]
    return f'{stem}_{flag}_sl{seq_len}_pl{pred_len}_fl{future_len}_clip_text.npy'


def clip_cache_path(cache_dir, data_path, flag, seq_len, pred_len, future_len):
    return os.path.join(cache_dir, clip_cache_name(
        data_path, flag, seq_len, pred_len, future_len))
