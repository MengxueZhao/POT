import torch
import argparse

PAD_idx = 0
UNK_idx = 1
BOS_idx = 2
EOS_idx = 3
SEP_idx = 4
CLSU_idx = 5
CLSP_idx = 6
CLSUA_idx = 7
CLSPA_idx = 8

begin = 9
aspect_token = {}  # aspect: [9, 37)
for i in range(28):
    aspect_token[i+begin] = 'ASP' + str(i)
end = begin + len(aspect_token)

aspect_token2name = {'ASP0': '[交通]', 'ASP1': '[繁华]', 'ASP2': '[显眼]', 'ASP3': '[排队]', 'ASP4': '[服务]',
                     'ASP5': '[停车]', 'ASP6': '[上菜]', 'ASP7': '[价格]', 'ASP8': '[性价比]', 'ASP9': '[折扣]',
                     'ASP10': '[装修]', 'ASP11': '[嘈杂]', 'ASP12': '[空间]', 'ASP13': '[卫生]', 'ASP14': '[分量]',
                     'ASP15': '[口感]', 'ASP16': '[外观]', 'ASP17': '[推荐]', 'ASP18': '[感受]', 'ASP19': '[再来]',
                     'ASP20': '[食材]', 'ASP21': '[品种]', 'ASP22': '[氛围]', 'ASP23': '[特色]', 'ASP24': '[正宗]',
                     'ASP25': '[创新]', 'ASP26': '[锅底]', 'ASP27': '[辣度]'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_opts(opts):
    """
    Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def parser_parameter():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name', type=str, default="EED", help='Model.')
    parser.add_argument(
        "--do_train", type=int, default=1, help="Do train or test.")
    parser.add_argument(
        '--gpu_id', type=str, default="0", help='GPU id.')

    parser.add_argument(
        '--dataset_ratio', type=str, default="80,10,10", help='Dataset segmentation ratio. (0, 100)')
    parser.add_argument(
        '--shuffle', action='store_true', default=False, help="Shuffle dataset.")

    parser.add_argument(
        '--mini_data', action='store_true', default=False, help="Sample mini graph for training.")
    parser.add_argument(
        '--mini_data_range', type=str, default="75,23,256,25,512,1500,9", help="Sample/Cut: maximum number of"
                                                                               "POIs connected by the User"
                                                                               "aspect connected by the User"
                                                                               "Words connected by the Aspect of User"
                                                                               "aspect connected by the POI"
                                                                               "Words connected by the Aspect of POI"
                                                                               "max length of input poi words sequence"
                                                                               "and max length of label tag sequence.")

    parser.add_argument(
        "--pos_neg_sample", type=int, default=30, help="Number of positive/negative samples in Contrastive Learning.")
    parser.add_argument(
        '--data_dir', type=str, default='./DataSet', help="Data file directory.")
    parser.add_argument(
        '--pkl_dir', type=str, default='./Output/pkl_files', help="Checkpoint directory.")

    parser.add_argument(
        '--batch', type=int, default=4, help="Batch size")
    parser.add_argument(
        "--seed", type=int, default=99, help="Set the random seed of the experiment.")

    parser.add_argument(
        "--feature_dim", type=str, default="200,256,256,256", help="Word embedding/Aspect/POI/User dim.")
    parser.add_argument(
        '--enc_gat_size', type=str, default="2,2", help='GAT encoder multi-head/layer number.')
    parser.add_argument(
        '--enc_trans_size', type=str, default="2,2", help='Transformer encoder multi-head/layer number.')
    parser.add_argument(
        '--dec_trans_size', type=str, default="2,2", help='Transformer decoder multi-head/layer number.')

    parser.add_argument(
        "--depth", type=int, default=40, help='key/value depth in Transformer decoder.')
    parser.add_argument(
        "--hidden_size", type=int, default=256, help='Transformer decoder: PositionWiseFeedForward：'
                                                     'Linear + RELU + Linear, size: hidden-filter-hidden.'
                                                     'must be equal to poi_dim.')
    parser.add_argument(
        "--filter", type=int, default=50, help='Transformer decoder: PositionWiseFeedForward：'
                                               'Linear + RELU + Linear, size: hidden-filter-hidden.')

    parser.add_argument(
        "--lr", type=float, default=0.0001, help='Learning rate for Adam.')

    parser.add_argument(
        "--epoch_number", type=int, default=15, help='Epoch number for training.')
    parser.add_argument(
        "--print_seg", type=int, default=960, help='Interval batch number printed during training.')

    parser.add_argument(
        "--ngram_size", type=int, default=2, help='Generate no repeat n_gram words when test.')

    parser.add_argument(
        "--behavior_upper_limit", type=str, default="15,3,3", help='Maximum number of clicks/favor/buy made by '
                                                                   'the user on the POI.')
    parser.add_argument(
        "--gcn_size", type=str, default="2,64,2,2", help='GCN layer number and GCN user/poi id embedding size and '
                                                         'GAT head number and GAT layer number.')
    parser.add_argument(
        "--similar_neighbor", type=int, default=15, help="The number of user similar neighbors.")
    parser.add_argument(
        "--top_number", type=int, default=6, help="Aspect top number for rank.")

    args = parser.parse_args()

    print_opts(args)

    return args