import torch
import sklearn
from tqdm import tqdm
import os

from Utils import config
from Utils.data_reader import read_source_data
from Utils.print_info import print_data_pair, print_data_ratio
import numpy as np
import random
import scipy.sparse as sp


def pad(ori_arr, pad_value, desired_num, padding_mode='r'):

    assert desired_num > 0
    if padding_mode == 'r':
        result = ori_arr[:desired_num] + [pad_value] * (desired_num - len(ori_arr))
    elif padding_mode == 'l':
        result = [pad_value] * (desired_num - len(ori_arr)) + ori_arr[-desired_num:]
    else:
        result = ori_arr[:desired_num]
    assert len(result) == desired_num
    return result


def get_aspect_word_matrix(vocab, history, max_a, max_aw):

    naw = []  # max_pa_length * max_paw_length
    for aspect in history:  # {a1:[w1,w2,w3], a2:[w1,w2,w3]}
        aw = []
        for word in history[aspect]:
            aw.append(vocab.word2index[word] if word in vocab.word2index else config.UNK_idx)
        padded_aw = pad(aw, config.PAD_idx, max_aw)
        naw.append(padded_aw)
        if len(naw) == max_a:
            break

    while len(naw) < max_a:  # pad 0
        naw.append([0] * max_aw)

    return naw


def get_sparse_indices_value(behavior_up, sparse_lists, uindex, poi_id2index, upper_limit):
    indices_row, indices_col, values = sparse_lists
    behavior_pois = []
    for tup in behavior_up:
        pindex = poi_id2index[tup[0]]-1
        number = tup[1] if tup[1] < upper_limit else upper_limit  # [0,1,0,14,0,…]

        if number > 0:
            indices_row.append(uindex)
            indices_col.append(pindex)
            values.append(number)

            behavior_pois.append(pindex)

    sparse_lists = (indices_row, indices_col, values)

    return sparse_lists, behavior_pois


def get_sparse_tensor(sparse_lists, user_number, poi_number):
    indices_row, indices_col, values = sparse_lists

    indices = []
    indices.append(indices_row)
    indices.append(indices_col)
    indices = torch.tensor(indices, dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float)
    
    # adjacency matrix normalization
    sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size([user_number, poi_number]))
    A = sparse_tensor  # user_number * poi_number
    D = torch.sparse.sum(A, dim=1) ** (-0.5)  # user_number * 1
    # diagonal matrix，=> user_number * user_number
    dig_D = torch.tensor(sp.diags(diagonals=D.to_dense().unsqueeze(0), offsets=[0]).todense(), dtype=torch.float)
    D_A = torch.sparse.mm(A.t(), dig_D).t()  # user_number * poi_number

    adj = D_A.numpy()
    new_indices = torch.Tensor(adj.nonzero()).long()
    new_values = torch.Tensor(adj[adj.nonzero()]).squeeze(0)

    sparse_adj = torch.sparse.FloatTensor(new_indices, new_values, torch.Size([user_number, poi_number]))

    return sparse_adj  # user_number * poi_number


def get_behavior_matrix(behavior_graph, user_id2index, poi_id2index, behavior_range, max_up_length):

    upper_limit, id_embed_size = behavior_range
    click_limit, favor_limit, consume_limit = upper_limit

    click_indices_row, click_indices_col, click_values = [], [], []
    click_sparse_lists = (click_indices_row, click_indices_col, click_values)
    favor_indices_row, favor_indices_col, favor_values = [], [], []
    favor_sparse_lists = (favor_indices_row, favor_indices_col, favor_values)
    consume_indices_row, consume_indices_col, consume_values = [], [], []
    consume_sparse_lists = (consume_indices_row, consume_indices_col, consume_values)

    behavior_samples_matrix = [[0] * (max_up_length * 2)] * len(user_id2index)  # user_number * 24
    behavior_labels_matrix = [[0] * (max_up_length * 2)] * len(user_id2index)
    for uid in behavior_graph:
        uindex = user_id2index[uid] - 1

        behavior_pois = []
        click_up = behavior_graph[uid]['click']  # [(pid,number),(),()]
        click_sparse_lists, click_behavior_pois = get_sparse_indices_value(click_up, click_sparse_lists,
                                                                           uindex, poi_id2index, click_limit)
        favor_up = behavior_graph[uid]['favor']
        if len(favor_up) > 0:
            favor_sparse_lists, favor_behavior_pois = get_sparse_indices_value(favor_up, favor_sparse_lists,
                                                                               uindex, poi_id2index, favor_limit)
        else:
            favor_behavior_pois = []

        consume_up = behavior_graph[uid]['consume']
        if len(consume_up) > 0:
            consume_sparse_lists, consume_behavior_pois = get_sparse_indices_value(consume_up, consume_sparse_lists,
                                                                                   uindex, poi_id2index, consume_limit)
        else:
            consume_behavior_pois = []

        # behavior sample
        behavior_sample_number = max_up_length * 2 + max_up_length // 3  # positive:12, negative:12, hard negative:4
        behavior_pois.extend(click_behavior_pois)
        behavior_pois.extend(favor_behavior_pois)
        behavior_pois.extend(consume_behavior_pois)
        input_behavior_pois = behavior_pois.copy()

        review_pois = [poi_id2index[pid] - 1 for pid in behavior_graph[uid]['review']]
        behavior_pois.extend(review_pois)
        negative_list = list(set(range(len(poi_id2index))) - set(behavior_pois))  # behavior: none, label: 0
        hard_negative_list = list(set(input_behavior_pois) - set(review_pois))  # behavior: yes, label: 0

        behavior_up_index = review_pois[:max_up_length]
        if len(behavior_up_index) <= 6:  # Restricted minimum positive examples
            positive_samples = np.random.choice(behavior_up_index, 6, replace=True).tolist()
        else:
            positive_samples = behavior_up_index.copy()

        positive_number = len(positive_samples)

        hard_negative_number = positive_number // 3
        hard_negative_samples = np.random.choice(hard_negative_list, hard_negative_number, replace=False).tolist()

        negative_number = behavior_sample_number - positive_number - hard_negative_number
        negative_samples = np.random.choice(negative_list, negative_number, replace=False).tolist()

        behavior_samples = []  # positive + negative (poi index)
        behavior_samples.extend(positive_samples)
        behavior_samples.extend(hard_negative_samples)
        behavior_samples.extend(negative_samples)

        # labels
        behavior_labels = [1] * positive_number
        behavior_labels.extend([0] * (behavior_sample_number - positive_number))

        # shuffle
        samples = list(zip(behavior_samples, behavior_labels))
        random.shuffle(samples)
        behavior_samples[:], behavior_labels[:] = zip(*samples)

        behavior_samples_matrix[uindex] = behavior_samples
        behavior_labels_matrix[uindex] = behavior_labels

    del behavior_graph
    behavior_samples_matrix = torch.tensor(behavior_samples_matrix, dtype=torch.long)
    behavior_labels_matrix = torch.tensor(behavior_labels_matrix, dtype=torch.long)
    behavior_data_matrix = (behavior_samples_matrix, behavior_labels_matrix)

    click_sparse_graph = get_sparse_tensor(click_sparse_lists, len(user_id2index), len(poi_id2index))
    del click_indices_row, click_indices_col, click_values, click_sparse_lists
    favor_sparse_graph = get_sparse_tensor(favor_sparse_lists, len(user_id2index), len(poi_id2index))
    del favor_indices_row, favor_indices_col, favor_values, favor_sparse_lists
    consume_sparse_graph = get_sparse_tensor(consume_sparse_lists, len(user_id2index), len(poi_id2index))
    del consume_indices_row, consume_indices_col, consume_values, consume_sparse_lists

    sparse_graphs = (click_sparse_graph, favor_sparse_graph, consume_sparse_graph)

    # Initialize id matrix
    uid_embed = torch.normal(mean=0, std=torch.zeros(len(user_id2index), id_embed_size).fill_(0.05))  # 0.005
    pid_embed = torch.normal(mean=0, std=torch.zeros(len(poi_id2index), id_embed_size).fill_(0.05))

    behavior_matrix = (sparse_graphs, uid_embed, pid_embed)

    return behavior_matrix, behavior_data_matrix


def pre_process(data_pair, vocab, src_dict, mini_range, behavior_range):

    user_history_aw_dict, poi_history_aw_dict, positive_pu_dict, negative_pu_dict, behavior_graph = src_dict

    max_up_length, max_ua_length, max_uaw_length, max_pa_length, max_paw_length, \
    max_pw_length, max_label_length, pos_neg_sample, top_number = mini_range

    # Construct aspect-word matrix
    uaw_matrix, paw_matrix = [], []  # user/poi_number * aspect_number * uaw/paw
    pu_matrix = []  # poi_number * pos_neg_sample*2
    user_id2index, poi_id2index = {}, {}
    user_index, poi_index = 1, 1  # poi/user index from 1, 0 is used to pad

    # Construct user-aspect-word matrix and mask matrix, user id2index, word padding
    uaw_matrix.append([[0]*max_uaw_length]*max_ua_length)  # 1 * max_ua_length * max_uaw_length

    for uid in user_history_aw_dict:
        user_id2index[uid] = user_index
        user_index += 1
        uaw = get_aspect_word_matrix(vocab, user_history_aw_dict[uid], max_ua_length, max_uaw_length)
        uaw_matrix.append(uaw)
    del user_history_aw_dict

    # Construct poi-aspect-word matrix and mask matrix, poi id2index, word padding, construct poi-user matrix (for CL)
    paw_matrix.append([[0]*max_paw_length]*max_pa_length)  # 1 * max_pa_length * max_paw_length
    pu_matrix.append([0] * (pos_neg_sample * 2))  # 1 * pos_neg_sample+pos_neg_sample

    for pid in poi_history_aw_dict:
        poi_id2index[pid] = poi_index
        poi_index += 1
        paw = get_aspect_word_matrix(vocab, poi_history_aw_dict[pid], max_pa_length, max_paw_length)
        paw_matrix.append(paw)

        pu = []
        positive_user = [user_id2index[uid] for uid in positive_pu_dict[pid]]
        padded_positive_user = pad(positive_user, config.PAD_idx, pos_neg_sample)
        negative_user = [user_id2index[uid] for uid in negative_pu_dict[pid]]
        padded_negative_user = pad(negative_user, config.PAD_idx, pos_neg_sample)
        pu.extend(padded_positive_user)
        pu.extend(padded_negative_user)
        pu_matrix.append(pu)

    del poi_history_aw_dict
    del positive_pu_dict
    del negative_pu_dict

    uaw_matrix = torch.tensor(uaw_matrix, dtype=torch.long)
    paw_matrix = torch.tensor(paw_matrix, dtype=torch.long)
    pu_matrix = torch.tensor(pu_matrix, dtype=torch.long)

    Matrixes = (uaw_matrix, paw_matrix, pu_matrix)

    behavior_matrix, behavior_data_matrix = get_behavior_matrix(behavior_graph, user_id2index, poi_id2index,
                                                                behavior_range, max_up_length)
    behavior_samples_matrix, behavior_labels_matrix = behavior_data_matrix

    # Solve data_pair, up pid2pindex, pw word2wid, ugc word2wid, padding
    count = 0  # for behavior
    for single_data in tqdm(data_pair):
        input_user_uid = single_data['src_Uid']
        input_poi_pid = single_data['src_Pid']
        single_data['user_index'] = [user_id2index[input_user_uid]]  # index from 1
        del single_data['src_Uid']
        del single_data['src_Pid']

        input_user_up = single_data['src_Up']  # [pid1, pid2, pid3]
        input_user_up_index = [poi_id2index[pid] for pid in input_user_up]
        padded_input_pois = pad(input_user_up_index, config.PAD_idx, max_up_length)
        padded_input_pois.append(poi_id2index[input_poi_pid])  # add input to last dimension (padding + input)
        single_data['input_pois'] = padded_input_pois
        del single_data['src_Up']

        single_data['input_user_history'] = uaw_matrix[user_id2index[input_user_uid]].tolist()

        input_poi_words = []
        for w in single_data['src_Pw_seq']:
            input_poi_words.append(vocab.word2index[w] if w in vocab.word2index else config.UNK_idx)
        padded_input_poi_words = pad(input_poi_words, config.PAD_idx, max_pw_length)
        padded_input_poi_words[-1] = config.PAD_idx if padded_input_poi_words[-1] == config.PAD_idx else config.EOS_idx

        single_data['input_poi_words'] = padded_input_poi_words
        del single_data['src_Pw_seq']

        ugc_tags_label = []  # top_number * max_label_length
        for rank in single_data['src_label_tags']:
            if rank >= top_number:
                break
            aspect_tags = []
            for w in single_data['src_label_tags'][rank]:
                aspect_tags.append(vocab.word2index[w] if w in vocab.word2index else config.UNK_idx)
            padded_aspect_tags = pad(aspect_tags, config.PAD_idx, max_label_length)
            padded_aspect_tags[-1] = config.PAD_idx if padded_aspect_tags[-1] == config.PAD_idx else config.EOS_idx
            ugc_tags_label.append(padded_aspect_tags)
        while len(ugc_tags_label) < top_number:  # EOS PAD PAD PAD PAD PAD
            ugc_tags_label.append([config.EOS_idx] + ([config.PAD_idx] * (max_label_length - 1)))
        single_data['output_ugc_label'] = ugc_tags_label
        del single_data['src_label_tags']

        single_data['output_rank_label'] = single_data['src_label_rank']
        del single_data['src_label_rank']

        # for behavior align update
        update_number = 24  # hyperparameter: 24
        behavior_user = torch.tensor(list(range(0, update_number)))
        behavior_user = ((behavior_user + count * update_number) % len(user_id2index)).tolist()  # length: 25
        total_behavior_user = [user_id2index[input_user_uid] - 1]  # behavior index-1
        total_behavior_user.extend(behavior_user)
        single_data['input_user_index'] = total_behavior_user
        count += 1

        # store pois corresponding to the user who should be taken out to calculate loss in UP matrix
        single_data['input_behavior_poi_index'] = behavior_samples_matrix[torch.tensor(total_behavior_user,
                                                                                       dtype=torch.long)].tolist()
        single_data['input_behavior_labels'] = behavior_labels_matrix[torch.tensor(total_behavior_user,
                                                                                   dtype=torch.long)].tolist()
    print_data_pair(data_pair)

    return data_pair, Matrixes, behavior_matrix


def save_dataset(para, data_pair, vocab, vector_dict, src_dict):

    user_history_aw_dict, poi_history_aw_dict, _, _, _ = src_dict
    _, embed_size, _, _ = [int(x) for x in para.gcn_size.strip().split(",")]

    click_limit, favor_limit, consume_limit = [int(x) for x in para.behavior_upper_limit.strip().split(",")]
    behavior_upper_limit = (click_limit, favor_limit, consume_limit)

    max_up_length, max_ua_length, max_uaw_length, max_pa_length, max_paw_length, \
    max_pw_length, max_label_length = [int(x) for x in para.mini_data_range.strip().split(",")]

    mini_range = (max_up_length, max_ua_length, max_uaw_length, max_pa_length, max_paw_length, max_pw_length,
                  max_label_length, para.pos_neg_sample, para.top_number)

    for single_data in data_pair:
        user_up = single_data['src_Up']
        if len(user_up) > max_up_length:
            user_up = user_up[len(user_up) - max_up_length:]
            single_data['src_Up'] = user_up

    print_data_ratio(data_pair, mini_range, user_history_aw_dict, poi_history_aw_dict)
    behavior_range = (behavior_upper_limit, embed_size)

    if para.shuffle:
        data_pair = sklearn.utils.shuffle(data_pair, random_state=para.seed)

    data_pair, Matrixes, behavior_matrix = pre_process(data_pair, vocab, src_dict, mini_range, behavior_range)

    D = (data_pair, vocab, vector_dict)
    M = (Matrixes, behavior_matrix)
    torch.save(D, os.path.join(para.data_dir, 'data.pkl'), pickle_protocol=4)
    torch.save(M, os.path.join(para.data_dir, 'matrix.pkl'), pickle_protocol=4)


def get_dataset(para):
    data_pair, vocab, vector_dict, src_dict = read_source_data(para)
    save_dataset(para, data_pair, vocab, vector_dict, src_dict)