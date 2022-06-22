import os
from tqdm import tqdm
import numpy as np
import torch

from Utils import config


class Lang:
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


def solve_behavior_data(data_dir, User_POI_dict):

    # Behavior (Input data)
    Click_dict = np.load(os.path.join(data_dir, 'Source_Data/Click_UP.npy'), allow_pickle=True).item()
    Favor_dict = np.load(os.path.join(data_dir, 'Source_Data/Favor_UP.npy'), allow_pickle=True).item()
    Consume_dict = np.load(os.path.join(data_dir, 'Source_Data/Buy_UP.npy'), allow_pickle=True).item()

    behavior_graph = {}
    for uid in User_POI_dict:
        if uid not in Click_dict:
            print("ERROR! click can not cover total user.")
        Favor_dict[uid] = [] if uid not in Favor_dict else Favor_dict[uid]
        Consume_dict[uid] = [] if uid not in Consume_dict else Consume_dict[uid]

        behavior_graph[uid] = {'click': Click_dict[uid], 'favor': Favor_dict[uid], 'consume': Consume_dict[uid],
                               'review': User_POI_dict[uid]}

    return behavior_graph


def solve_cl_data(data_dir, pos_neg_sample, User_POI_dict):

    # CL (Input data)
    POI_User_dict = np.load(os.path.join(data_dir, 'Source_Data/POI_User.npy'), allow_pickle=True).item()

    positive_pu_dict, negative_pu_dict = {}, {}
    for pid in POI_User_dict:
        positive_pu = POI_User_dict[pid]
        if len(positive_pu) > pos_neg_sample:
            positive_pu_dict[pid] = np.random.choice(positive_pu, pos_neg_sample, replace=False)
        else:
            positive_pu_dict[pid] = positive_pu

        negative_pu = []
        for uid in User_POI_dict:
            if uid not in positive_pu:
                negative_pu.append(uid)
        if len(negative_pu) > pos_neg_sample:
            negative_pu_dict[pid] = np.random.choice(negative_pu, pos_neg_sample, replace=False)
        else:
            negative_pu_dict[pid] = negative_pu

    return positive_pu_dict, negative_pu_dict


def read_source_data(para):
    # input data
    User_POI_dict = np.load(os.path.join(para.data_dir, 'Source_Data/User_POI_rank.npy'), allow_pickle=True).item()
    UGC_Aspect_Word_dict = np.load(os.path.join(para.data_dir, 'Source_Data/UGC_Aspect_Word.npy'), allow_pickle=True).item()
    User_PR_dict = np.load(os.path.join(para.data_dir, 'Source_Data/User_PR.npy'), allow_pickle=True).item()
    # POI_Info_dict = np.load(os.path.join(para.data_dir, 'POI_Info.npy'), allow_pickle=True).item()
    # output data
    UGC_RankedTags_dict = np.load(os.path.join(para.data_dir, 'Source_Data/UGC_RankedTags.npy'), allow_pickle=True).item()
    UGC_RankedAspects_dict = np.load(os.path.join(para.data_dir, 'Source_Data/UGC_RankedAspects.npy'), allow_pickle=True).item()
    UGC_UP_dict = np.load(os.path.join(para.data_dir, 'Source_Data/UGC_UP.npy'), allow_pickle=True).item()

    # Construct WAP data of WAPU-meta path and WAU data of WAU-meta path in HHRG
    user_history_aw_dict, poi_history_aw_dict = {}, {}
    for uid in User_PR_dict:
        user_history_aw_dict[uid] = {}
        # user_total_words_number = {}
        for pr_tuple in User_PR_dict[uid]:
            rid = pr_tuple[1]

            pid = pr_tuple[0]
            if pid not in poi_history_aw_dict:
                poi_history_aw_dict[pid] = {}

            for a in UGC_Aspect_Word_dict[rid]:
                if a not in user_history_aw_dict[uid]:
                    user_history_aw_dict[uid][a] = []
                user_history_aw_dict[uid][a].extend(UGC_Aspect_Word_dict[rid][a])

                if a not in poi_history_aw_dict[pid]:
                    poi_history_aw_dict[pid][a] = []
                poi_history_aw_dict[pid][a].extend(UGC_Aspect_Word_dict[rid][a])

    positive_pu_dict, negative_pu_dict = solve_cl_data(para.data_dir, para.pos_neg_sample, User_POI_dict)
    behavior_graph = solve_behavior_data(para.data_dir, User_POI_dict)

    src_dict = (user_history_aw_dict, poi_history_aw_dict, positive_pu_dict, negative_pu_dict, behavior_graph)

    # Construct data pair
    data_pair, input_words_appear_number = [], 0
    vocab = Lang()
    for rid in tqdm(UGC_UP_dict):
        uid = UGC_UP_dict[rid][0]
        pid = UGC_UP_dict[rid][1]

        single_data = {}
        single_data['src_Uid'] = uid
        single_data['src_Pid'] = pid
        single_data['src_Rid'] = rid
        single_data['src_Up'] = User_POI_dict[uid]  # user reviewed poi

        # Statistics word frequency
        for user_pid in User_POI_dict[uid]:  # input_user poi_list
            for user_pid_aspect in poi_history_aw_dict[user_pid]:
                vocab.index_words(poi_history_aw_dict[user_pid][user_pid_aspect])
                input_words_appear_number += len(poi_history_aw_dict[user_pid][user_pid_aspect])
        for pid_aspect in poi_history_aw_dict[pid]:  # input_poi
            vocab.index_words(poi_history_aw_dict[pid][pid_aspect])
            input_words_appear_number += len(poi_history_aw_dict[pid][pid_aspect])

        label_tags_dict = {}  # {0: [ASP, w1, w2, EOS], 1: [ASP, w1, w2, EOS]} key: rank，value: asp+word
        label_rank = torch.zeros([len(config.aspect_token)], dtype=torch.long)
        for aw_index in range(len(UGC_RankedTags_dict[rid])):
            aspect_tags = []
            word_list = UGC_RankedTags_dict[rid][aw_index]
            aspect_id = UGC_RankedAspects_dict[rid][aw_index]
            aspect_tags.append(config.aspect_token[aspect_id + config.begin])  # ASP
            aspect_tags.extend(word_list)  # word
            aspect_tags.append("EOS")
            label_tags_dict[aw_index] = aspect_tags
            label_rank[aspect_id] = aw_index + 1  # rank
            vocab.index_words(word_list)
            input_words_appear_number += len(word_list)
        single_data['src_label_tags'] = label_tags_dict

        label_score = max(label_rank) + 1 - label_rank
        final_label_score = (label_score < max(label_rank) + 1) * label_score
        single_data['src_label_rank'] = final_label_score.tolist()

        poi_sequence_words = []
        for aspect in poi_history_aw_dict[pid]:
            poi_sequence_words.append(config.aspect_token[aspect + config.begin])
            poi_sequence_words.extend(poi_history_aw_dict[pid][aspect])
        poi_sequence_words.append("EOS")

        # Transformer input
        single_data['src_Pw_seq'] = poi_sequence_words  # [t1_word1, t1_word2, 'ASP', t2_word, 'ASP', t3_word, 'EOS']

        data_pair.append(single_data)

    # filter vocab words
    w2c = dict(sorted(vocab.word2count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    # default tokens
    tokens = {config.PAD_idx: "PAD", config.UNK_idx: "UNK", config.BOS_idx: "BOS", config.EOS_idx: "EOS",
              config.SEP_idx: "SEP", config.CLSU_idx: "CLSU", config.CLSP_idx: "CLSP",
              config.CLSUA_idx: "CLSUA", config.CLSPA_idx: "CLSPA"}
    tokens.update(config.aspect_token)
    vocab.add_funs(tokens)
    dict_words_appear_number = 0
    for w in w2c:
        if w2c[w] <= 150:
            break
        vocab.word2index[w] = vocab.n_words
        vocab.index2word[vocab.n_words] = w
        vocab.n_words += 1
        dict_words_appear_number += w2c[w]

    print("vocab_size:", vocab.n_words)
    print("ratio: vocab_size | total_words:", vocab.n_words / len(w2c))
    print("ratio: vocab_words_appear_number | total_words_appear_number:", dict_words_appear_number / input_words_appear_number)

    # Construct word2vector
    # Download from https://ai.tencent.com/ailab/nlp/en/embedding.html
    # word_vector_Tencent = open(os.path.join(para.data_dir, 'Tencent_AILab_ChineseEmbedding.txt'), 'r', encoding='utf-8')
    word_vector_Tencent = open('/home/hadoop-aipnlp/cephfs/data/zhaomengxue/2022SIGIR/Data/Tencent_AILab_ChineseEmbedding.txt', 'r', encoding='utf-8')
    vector_dict = {}
    for i, line in enumerate(word_vector_Tencent.readlines()):
        wv = line.rstrip('\n').split(" ")
        w = wv[0]
        if w in vocab.word2index:
            v = wv[1:]
            v = [float(value) for value in v]
            vector_dict[w] = v
    print('pretrained vector size: ', len(vector_dict))

    return data_pair, vocab, vector_dict, src_dict
