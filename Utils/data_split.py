import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from functools import partial
import os
import time

from Utils import config
from Utils.data_loader import get_dataset


class Lang:  # vocab
    def __init__(self):
        self.word2count = {}

    def add_funs(self, init_index2word):
        self.init_index2word = init_index2word
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.n_words = len(init_index2word)  # Count default tokens

        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.index2word = init_index2word

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1


class ChunkSampler(Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class InteractionDataSet(Dataset):
    def __init__(self, para, data_pair, Matrixes, behavior_matrix):
        self.max_up_length, self.max_ua_length, self.max_uaw_length, self.max_pa_length, self.max_paw_length, \
        self.max_pw_length, self.max_label_length = [int(x) for x in para.mini_data_range.strip().split(",")]

        self.mini_range = (self.max_up_length, self.max_ua_length, self.max_uaw_length, self.max_pa_length,
                           self.max_paw_length, self.max_pw_length, self.max_label_length,
                           para.pos_neg_sample, para.top_number)

        self.data_pair, self.Matrixes, self.behavior_matrix = data_pair, Matrixes, behavior_matrix

    def __getitem__(self, item):
        return self.data_pair[item]

    def get_total_data_number(self):
        return len(self.data_pair)


def collate_fn(batch_data):  # CPU => GPU

    input_pois = [single_data['input_pois'] for single_data in batch_data]  # user poi history + input poi
    input_pois = torch.tensor(input_pois, dtype=torch.long)

    # Transformer encoder input
    input_poi_words = [single_data['input_poi_words'] for single_data in batch_data]  # word sequence of input poi
    input_poi_words = torch.tensor(input_poi_words, dtype=torch.long)

    ugc_label = [single_data['output_ugc_label'] for single_data in batch_data]  # data label
    ugc_label = torch.tensor(ugc_label, dtype=torch.long)

    rank_label = [single_data['output_rank_label'] for single_data in batch_data]
    rank_label = torch.tensor(rank_label, dtype=torch.float)

    input_user_history = [single_data['input_user_history'] for single_data in batch_data]
    input_user_history = torch.tensor(input_user_history, dtype=torch.long)

    input_user_index = [single_data['input_user_index'] for single_data in batch_data]
    input_user_index = torch.tensor(input_user_index, dtype=torch.long)  # bs * 25

    input_behavior_poi_index = [single_data['input_behavior_poi_index'] for single_data in batch_data]
    input_behavior_poi_index = torch.tensor(input_behavior_poi_index, dtype=torch.long)  # bs * 25 * 24

    input_behavior_labels = [single_data['input_behavior_labels'] for single_data in batch_data]
    input_behavior_labels = torch.tensor(input_behavior_labels, dtype=torch.float)  # bs * 25 * 24

    user_index = [single_data['user_index'] for single_data in batch_data]
    user_index = torch.tensor(user_index, dtype=torch.long)  # bs * 1

    return {'input_pois': input_pois, 'input_user_history': input_user_history,
            'input_poi_words': input_poi_words, 'output_ugc_label': ugc_label, 'output_rank_label': rank_label,
            'input_user_index': input_user_index, 'input_behavior_poi_index': input_behavior_poi_index,
            'input_behavior_labels': input_behavior_labels, 'user_index': user_index}


def get_data_pairs(batch_data, matrixes):
    batch = collate_fn(batch_data)
    uaw_matrix, paw_matrix, pu_matrix = matrixes
    pos_neg_sample = pu_matrix.size(-1) // 2

    input_pois = batch['input_pois']  # bs * max_up_length+1

    # user history => input user + negative user
    total_pu = pu_matrix[input_pois][:, :, pos_neg_sample:]  # bs * max_up+1 * neg
    user_history = uaw_matrix[total_pu]  # bs * max_up_length+1 * neg * max_ua_length * max_uaw_length
    input_user_history = batch['input_user_history']  # bs * max_ua * max_uaw
    input_user_history = input_user_history.unsqueeze(1)  # bs * 1 * max_ua * max_uaw
    input_user_history = input_user_history.repeat([1, user_history.size(1), 1, 1]).unsqueeze(
        dim=2)  # bs * max_up+1 * 1 * max_ua * max_uaw
    user_history = torch.cat((input_user_history, user_history), dim=2)  # bs * max_up+1 * 1+neg * max_ua * max_uaw

    # cl users index
    user_index = batch['user_index'].unsqueeze(1).repeat([1, total_pu.size(1), 1])  # bs * max_up+1 * 1
    cl_user_index = torch.cat((user_index, total_pu), dim=-1)  # bs * max_up+1 * 1+neg
    cl_user_index = cl_user_index - 1  # global matrix index from 0

    # poi history
    poi_history = paw_matrix[input_pois]  # bs * max_up+1 * max_pa * max_paw

    input_poi_words = batch['input_poi_words']  # bs * max_pw_length
    ugc_label = batch['output_ugc_label']  # bs * max_label_length
    rank_label = batch['output_rank_label']
    user_index = batch['input_user_index']  # bs * 25
    input_behavior_poi_index = batch['input_behavior_poi_index']
    input_behavior_labels = batch['input_behavior_labels']

    pairs_data = (user_history, poi_history, input_pois, input_poi_words, ugc_label, rank_label, user_index,
                  input_behavior_poi_index, input_behavior_labels, cl_user_index)

    return pairs_data


def construct_data(para, n_gpu):

    if not os.path.exists(os.path.join(para.data_dir, 'data.pkl')):
        t1 = time.perf_counter()
        get_dataset(para)
        t2 = time.perf_counter()
        print("Data Loading Time:", t2 - t1)

    data_pair, vocab, vector_dict = torch.load(os.path.join(para.data_dir, 'data.pkl'))
    Matrixes, behavior_matrix = torch.load(os.path.join(para.data_dir, 'matrix.pkl'))

    interaction_dataset = InteractionDataSet(para, data_pair, Matrixes, behavior_matrix)

    N = interaction_dataset.get_total_data_number()

    dataset_ratio = [int(x) for x in para.dataset_ratio.strip().split(",")]
    train_start, valid_start, test_start = 0, \
                                           int(N * (dataset_ratio[0]) / 100), \
                                           int(N * (dataset_ratio[0] + dataset_ratio[1]) / 100)

    train_number = valid_start - train_start
    vaild_number = test_start - valid_start
    test_number = N - test_start

    collate_fnn = partial(get_data_pairs, matrixes=interaction_dataset.Matrixes)

    train_loader = DataLoader(interaction_dataset, batch_size=para.batch * n_gpu, num_workers=0,
                              sampler=ChunkSampler(train_number, 0), collate_fn=collate_fnn)

    valid_loader = DataLoader(interaction_dataset, batch_size=para.batch * n_gpu, num_workers=0,
                              sampler=ChunkSampler(vaild_number, valid_start), collate_fn=collate_fnn)

    test_loader = DataLoader(interaction_dataset, batch_size=para.batch * n_gpu, num_workers=0,
                             sampler=ChunkSampler(test_number, test_start), collate_fn=collate_fnn)

    print(f'total dataset size: {N} \n'
          f'train dataset size: {train_number}, batch number: {len(train_loader)}\n'
          f'valid dataset size: {vaild_number}, batch number: {len(valid_loader)}\n'
          f'test dataset size:  {test_number}, batch number: {len(test_loader)}.')

    dataset = (train_loader, valid_loader, test_loader)
    mini_range = interaction_dataset.mini_range
    behavior_matrix = interaction_dataset.behavior_matrix

    # to_device
    sparse_graphs, uid_embed, pid_embed = behavior_matrix
    click_sparse_graph, favor_sparse_graph, consume_sparse_graph = sparse_graphs

    click_sparse_graph = click_sparse_graph.to(config.device)
    favor_sparse_graph = favor_sparse_graph.to(config.device)
    consume_sparse_graph = consume_sparse_graph.to(config.device)

    uid_embed = uid_embed.to(config.device)  # uid_embed and pid_embed should be trainable parameters
    pid_embed = pid_embed.to(config.device)
    # poi_behavior_info = poi_behavior_info.to(config.device)

    sparse_graphs = (click_sparse_graph, favor_sparse_graph, consume_sparse_graph)
    behavior_matrix = (sparse_graphs, uid_embed, pid_embed)

    return vocab, vector_dict, dataset, mini_range, behavior_matrix