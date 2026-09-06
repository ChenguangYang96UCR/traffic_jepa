import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np
import os
import json


def infer_incident_cardinalities(args):
    """Use released mappings, falling back to all split samples."""
    mapping_specs = (
        ('incident_num_descriptions', 'desc_mapping.json', 'Description'),
        ('incident_num_types', 'type_mapping.json', 'Type'),
    )
    unresolved = []
    for arg_name, filename, feature_name in mapping_specs:
        if getattr(args, arg_name) is not None:
            continue
        path = os.path.join(args.root_path, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as handle:
                setattr(args, arg_name, max(len(json.load(handle)), 1))
        else:
            unresolved.append((arg_name, feature_name))

    need_position = args.incident_num_positions is None
    maxima = {name: -1 for name, _ in unresolved}
    position_max = -1
    if unresolved or need_position:
        for split in ('train', 'val', 'test'):
            filename = args.fremont_file_pattern.format(flag=split)
            path = os.path.join(args.root_path, filename)
            samples = np.load(path, allow_pickle=True)
            for sample in samples:
                features = sample['incident_features']
                for arg_name, feature_name in unresolved:
                    value = (features.get(feature_name, 0)
                             if isinstance(features, dict) else
                             features[1 if feature_name == 'Description' else 2])
                    maxima[arg_name] = max(maxima[arg_name], int(value))
                if need_position:
                    position_max = max(
                        position_max, int(sample['incident_position']))
        for arg_name, _ in unresolved:
            setattr(args, arg_name, maxima[arg_name] + 1)
        if need_position:
            args.incident_num_positions = position_max + 1

    for name in ('incident_num_descriptions', 'incident_num_types',
                 'incident_num_positions'):
        if getattr(args, name) < 1:
            raise ValueError(f'Could not infer a positive --{name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TopoJEPA')

    # basic config
    parser.add_argument('--seed', type=int, default=2026,
                        help='random seed for Python, NumPy, PyTorch, and CUDA')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='TopoJEPA',
                        help='model name, options: [TopoJEPA, iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--fremont_file_pattern', type=str,
                        default='incident_{flag}.npy',
                        help='Fremont split filename pattern under root_path')
    parser.add_argument('--fremont_traffic_feature', type=int, default=0,
                        help='traffic channel selected from Fremont x_data/y_data')
    parser.add_argument('--fremont_use_time_features', action='store_true',
                        help='also use Fremont time-of-day/day-of-week channels; '
                             'off means strictly traffic-only input')
    parser.add_argument('--incident', action='store_true',
                        help='enable IGSTGNN-style incident conditioning for Fremont JEPA')
    parser.add_argument('--incident_scale', type=float, default=1.0,
                        help='scale applied to the temporally decayed incident effect')
    parser.add_argument('--incident_sigma', type=float, default=1.0,
                        help='positive Gaussian decay width over forecast steps')
    parser.add_argument('--incident_num_descriptions', type=int, default=None,
                        help='Description embedding cardinality; inferred by default')
    parser.add_argument('--incident_num_types', type=int, default=None,
                        help='incident Type embedding cardinality; inferred by default')
    parser.add_argument('--incident_num_positions', type=int, default=None,
                        help='incident_position embedding cardinality; inferred by default')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='iterations between detailed branch-loss logs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--model_variant', type=str, default='original',
                        choices=['original', 'predictor', 'jepa'],
                        help='original, predictor, or TopoJEPA')
    parser.add_argument('--jepa_weight', type=float, default=0.0,
                        help='weight of JEPA loss; 0 makes jepa use the predictor-only training path')
    parser.add_argument('--topo_weight', type=float, default=0.0,
                        help='weight of variable-level H0 persistence-diagram Wasserstein loss')
    parser.add_argument('--text_weight', type=float, default=0.0,
                        help='weight of variable-level CLIP text-JEPA loss')
    parser.add_argument('--text_embed_dim', type=int, default=512,
                        help='dimension of cached CLIP text embeddings')
    parser.add_argument('--text_embedding_dir', type=str, default='',
                        help='directory containing precomputed CLIP embeddings')
    parser.add_argument('--use_gnn', action='store_true',
                        help='fuse a graph encoder when an adjacency matrix is available')
    parser.add_argument('--adj_path', type=str, default='',
                        help='square adjacency matrix in labeled CSV or .npy format')
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--alignment_weight', type=float, default=0.0,
                        help='weight of Equation-6 Cramer alignment between GNN and Transformer tokens')
    parser.add_argument('--ema_momentum', type=float, default=0.996,
                        help='EMA momentum for the JEPA target encoder')
    parser.add_argument('--stgcn_kernel_size', type=int, default=3)
    parser.add_argument('--stgcn_cheb_order', type=int, default=3)
    parser.add_argument('--stgcn_blocks', type=int, default=2)
    parser.add_argument('--stgcn_temporal_channels', type=int, default=64)
    parser.add_argument('--stgcn_spatial_channels', type=int, default=16)
    parser.add_argument('--stgcn_output_channels', type=int, default=64)
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    args = parser.parse_args()

    if args.incident:
        if args.data != 'Fremont' or args.model != 'TopoJEPA':
            parser.error('--incident currently requires --data Fremont --model TopoJEPA')
        if args.exp_name == 'partial_train':
            parser.error('--incident is not supported with --exp_name partial_train')
        if args.text_weight > 0:
            parser.error('--incident and --text_weight cannot share the optional batch slot')
        if args.incident_sigma <= 0:
            parser.error('--incident_sigma must be positive')
        infer_incident_cardinalities(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    def build_setting(iteration):
        setting = (
            f'{args.model_id}_{args.model}_{args.data}'
            f'_ft{args.features}_sl{args.seq_len}_ll{args.label_len}'
            f'_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}'
            f'_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}'
            f'_fc{args.factor}_eb{args.embed}_dt{args.distil}'
            f'_{args.des}_seed{args.seed}'
            f'_{args.class_strategy}_{args.model_variant}'
            f'_jw{args.jepa_weight}_tw{args.topo_weight}'
            f'_textw{args.text_weight}_aw{args.alignment_weight}'
            f'_gnn{int(args.use_gnn)}'
        )
        if args.incident:
            setting += (f'_incident_s{args.incident_sigma}'
                        f'_scale{args.incident_scale}')
        return f'{setting}_{iteration}'

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = build_setting(ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = build_setting(ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
