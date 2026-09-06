from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_Fremont_NPY
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'Fremont': Dataset_Fremont_NPY,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len,
              max(args.seq_len, args.pred_len)],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    if Data is Dataset_Custom:
        data_kwargs['text_embedding_dir'] = getattr(args, 'text_embedding_dir', '')
    elif Data is Dataset_Fremont_NPY:
        data_kwargs['traffic_feature'] = getattr(
            args, 'fremont_traffic_feature', 0)
        data_kwargs['use_time_features'] = getattr(
            args, 'fremont_use_time_features', False)
        data_kwargs['use_incident'] = getattr(args, 'incident', False)
        data_kwargs['file_pattern'] = getattr(
            args, 'fremont_file_pattern', 'incident_{flag}.npy')
    data_set = Data(**data_kwargs)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
