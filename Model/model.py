import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from Utils import config
from Utils.embed import get_word_embedding, get_word_position_embedding
from Utils.embed import get_bias_mask, get_attn_subsequent_mask

from Model.gat import GAT_Graph
from Model.transformers import EncoderLayer, DecoderLayer, LayerNorm
from Model.generate import NoRepeatNGramLogitsProcessor
from Model.behavior import Behavior_Module
from Model.rank import get_rank_loss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Graph_Encoder(nn.Module):
    def __init__(self, feature_dim, graph_data_range, enc_gat_size):
        super(Graph_Encoder, self).__init__()

        word_dim, self.aspect_dim, _, self.user_dim = feature_dim
        heads_number, self.num_layer = enc_gat_size

        self.graph_layer_stack = nn.ModuleList()
        for i in range(self.num_layer):
            self.graph_layer_stack.append(GAT_Graph(heads_number, feature_dim, graph_data_range))

    def forward(self, inputs_feature, inputs_mask, prior_feature, only_use_user_history):

        # user_history_feature, poi_history_feature = inputs_feature
        # user_history_mask, poi_history_mask = inputs_mask
        # prior_user_aspect_feature, prior_poi_aspect_feature, prior_user_feature = prior_feature

        for i, gat_layer in enumerate(self.graph_layer_stack):
            prior_feature, cl_feature = gat_layer(inputs_feature, prior_feature, inputs_mask, only_use_user_history)

        _, _, input_user_feature = prior_feature  # bs * user_dim
        poi_history_feature, user_history_feature = cl_feature
        # bs * poi_number+1 * poi_dim
        # bs * poi_number+1 * user_number*2+1 * user_dim
        input_poi_feature = poi_history_feature[:, -1, :]  # bs * poi_dim

        return input_user_feature, input_poi_feature, cl_feature


class Trans_Encoder(nn.Module):
    def __init__(self, feature_dim, enc_trans_size, max_pw_length,
                 filter_size, total_key_depth, total_value_depth, use_mask=False,
                 input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):

        super(Trans_Encoder, self).__init__()

        word_dim, _, poi_dim, _ = feature_dim
        head_number, self.layer_number = enc_trans_size

        params = (poi_dim,
                  total_key_depth or poi_dim,
                  total_value_depth or poi_dim,
                  filter_size,
                  head_number,
                  get_bias_mask(max_pw_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(self.layer_number)])  # 是encoder的集合

        self.timing_signal = get_word_position_embedding(max_pw_length, poi_dim)

        self.embedding_proj = nn.Linear(word_dim, poi_dim, bias=False)
        self.layer_norm = LayerNorm(poi_dim)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):  # inputs: bs * max_pw_length * word_dim

        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)  # bs * max_pw_length * poi_dim

        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        for i in range(self.layer_number):
            x = self.enc[i](x, mask)

        y = self.layer_norm(x)

        return y


class Decoder(nn.Module):
    """
        A Transformer Decoder module.
        Inputs should be in the shape [batch_size, length, hidden_size]
        Outputs will have the shape [batch_size, length, hidden_size]
        Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, word_dim, hidden_size, filter_size, dec_trans_size,
                 total_key_depth, total_value_depth, max_length=100,
                 input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):

        super(Decoder, self).__init__()

        head_number, layer_number = dec_trans_size

        self.timing_signal = get_word_position_embedding(max_length, hidden_size)
        self.mask = Parameter(get_attn_subsequent_mask(max_length).float(), requires_grad=False)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  head_number,
                  get_bias_mask(max_length),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.dec = nn.Sequential(*[DecoderLayer(*params) for _ in range(layer_number)])

        self.embedding_proj = nn.Linear(word_dim, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output=None, mask=None):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)

        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        # Add timing signal
        # inputs: bs * label_length+1 * word_dim
        x += self.timing_signal[:, :inputs.size(1), :].type_as(inputs.data)

        # Run decoder
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

        # Final layer normalization
        y = self.layer_norm(y)

        return y, attn_dist


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, hidden_size, vocab_size, ngram_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.no_repeat_generate = NoRepeatNGramLogitsProcessor(ngram_size)
        self.vocab_size = vocab_size

    def forward(self, x, is_generate_aspect, batch_output_word, is_greedy):
        '''
           x: bs * length * hidden
        attn: bs * length * input_length
        '''

        pro_logit = self.proj(x)   # bs * length * vocab_size
        word_mask = torch.zeros([self.vocab_size], dtype=torch.long).to(config.device)
        word_mask[config.begin: config.end] = 1

        if is_generate_aspect:
            logit = pro_logit[:, :, config.begin: config.end].squeeze(dim=1)
        else:
            word_logit = pro_logit.masked_fill(word_mask.bool(), -1e18)
            logit = F.log_softmax(word_logit, dim=-1)  # bs * (1)length * vocab_size [2, 81, 43273]

            if is_greedy:
                logit = self.no_repeat_generate.get_scores(batch_output_word, logit[:, -1, :].squeeze(1))  # bs * vocab_size

        return logit


class cl_criterion(nn.Module):
    def __init__(self, user_dim, poi_dim):
        super(cl_criterion, self).__init__()

        self.cl_linear = nn.Linear(user_dim, poi_dim, bias=False)
        self.cl_loss = nn.CrossEntropyLoss()

    def forward(self, cl_feature):

        # poi_history_feature   # bs * poi_number+1 * poi_dim
        # user_history_feature  # bs * poi_number+1 * 1+user_number * user_dim
        poi_history_feature, user_history_feature = cl_feature
        # pos_neg_sample = user_history_feature.size(-2) - 1

        q = poi_history_feature.unsqueeze(2)      # bs * poi_number+1 * 1 * poi_dim
        k = self.cl_linear(user_history_feature)  # bs * poi_number+1 * 1+user_number * poi_dim

        negative_k = k[:, :, 1:, :]  # bs * poi_number+1 * pos_neg_sample * poi_dim
        score_neg = torch.matmul(q, negative_k.permute(0, 1, 3, 2))  # bs * poi_number+1 * 1 * pos_neg_sample

        positive_k = k[:, :, 0, :].unsqueeze(-1)  # bs * poi_number+1 * poi_dim * 1
        score_pos = torch.matmul(q, positive_k)  # bs * poi_number+1 * 1 * 1

        score_pos = score_pos.reshape(-1, score_pos.size(-1))  # bs*poi_number+1
        score_neg = score_neg.reshape(-1, score_neg.size(-1))  # bs*poi_number+1 * pos_neg_sample
        logits = torch.cat((score_pos, score_neg), dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(config.device)
        cl_loss = self.cl_loss(logits, labels)

        return cl_loss


class POT_Model(nn.Module):

    def __init__(self, para, vocab, vector_dict, mini_data_range, behavior_matrix):
        super(POT_Model, self).__init__()

        self.mini_data_range = mini_data_range
        max_up_length, max_ua_length, _, max_pa_length, _, max_pw_length, self.max_label_length, \
        pos_neg_sample, self.top_number = mini_data_range

        graph_data_range = (max_ua_length, max_pa_length, max_up_length)

        word_dim, aspect_dim, poi_dim, user_dim = [int(x) for x in para.feature_dim.strip().split(",")]
        self.feature_dim = (word_dim, aspect_dim, poi_dim, user_dim)

        enc_gat_heads_number, enc_gat_layer_number = [int(x) for x in para.enc_gat_size.strip().split(",")]
        self.enc_gat_size = (enc_gat_heads_number, enc_gat_layer_number)

        enc_trans_heads_number, enc_trans_layer_number = [int(x) for x in para.enc_trans_size.strip().split(",")]
        self.enc_trans_size = (enc_trans_heads_number, enc_trans_layer_number)

        dec_trans_heads_number, dec_trans_layer_number = [int(x) for x in para.dec_trans_size.strip().split(",")]
        self.dec_trans_size = (dec_trans_heads_number, dec_trans_layer_number)

        self.embedding = get_word_embedding(vocab, word_dim, vector_dict)
        self.graph_encoder = Graph_Encoder(self.feature_dim, graph_data_range, self.enc_gat_size)
        self.trans_encoder = Trans_Encoder(self.feature_dim, self.enc_trans_size, max_pw_length,
                                           para.filter, para.depth, para.depth)

        self.bos_linear = nn.Linear(poi_dim, word_dim, bias=False)
        self.decoder_input_linear = nn.Linear(word_dim + user_dim, word_dim, bias=False)

        self.decoder = Decoder(word_dim, para.hidden_size, para.filter, self.dec_trans_size, para.depth, para.depth)
        self.generator = Generator(para.hidden_size, vocab.n_words, para.ngram_size)

        self.generate_criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.cl_criterion = cl_criterion(user_dim, poi_dim)

        total_poi_number = behavior_matrix[0][0].size(1)
        self.behavior_module = Behavior_Module(para.gcn_size, para.similar_neighbor, max_up_length, total_poi_number)
        self.user_feature_linear = nn.Linear(user_dim*2, user_dim, bias=False)
        self.poi_title_feature_linear = nn.Linear(word_dim*10, word_dim, bias=False)

        # ======== Behavior Matrix ==========
        sparse_graphs, uid_embed, pid_embed = behavior_matrix
        cilck_sparse_graph, favor_sparse_graph, consume_sparse_graph = sparse_graphs

        self.cilck_sparse_graph = Parameter(cilck_sparse_graph, requires_grad=False)
        self.favor_sparse_graph = Parameter(favor_sparse_graph, requires_grad=False)
        self.consume_sparse_graph = Parameter(consume_sparse_graph, requires_grad=False)

        self.uid_embed = Parameter(uid_embed)
        self.pid_embed = Parameter(pid_embed)

    def get_init_feature(self, bs):  # init feature for GAT
        max_up_length, max_ua_length, _, max_pa_length, _, _, _, pos_neg_sample, _ = self.mini_data_range

        cls_ua_token = torch.LongTensor([config.CLSUA_idx] * bs).unsqueeze(1).to(config.device)  # (bsz, 1)
        cls_ua_feature = self.embedding(cls_ua_token)  # bs * 1 * word_dim
        cls_ua_feature = cls_ua_feature.unsqueeze(dim=1).unsqueeze(dim=1)
        user_history_aspect_init = cls_ua_feature.repeat([1, max_up_length+1, 1+pos_neg_sample, max_ua_length, 1])
        # bs * max_up_length+1 * 1+pos_neg_sample * max_ua_length * word_dim

        cls_pa_token = torch.LongTensor([config.CLSPA_idx] * bs).unsqueeze(1).to(config.device)  # (bsz, 1)
        cls_pa_feature = self.embedding(cls_pa_token)  # bs * 1 * word_dim
        cls_pa_feature = cls_pa_feature.unsqueeze(dim=1)  # bs * 1 * 1 * word_dim
        poi_history_aspect_init = cls_pa_feature.repeat([1, max_up_length + 1, max_pa_length, 1])
        # bs * max_up_length+1 * max_pa_length * word_dim

        init.xavier_uniform_(user_history_aspect_init)
        init.xavier_uniform_(poi_history_aspect_init)

        # bs * poi_number+1 * user_number*2+1 * aspect_number * aspect_dim
        # bs * poi_number+1 * aspect_number * aspect_dim
        init_feature = (user_history_aspect_init, poi_history_aspect_init, None)

        cls_user_token = torch.LongTensor([config.CLSU_idx] * bs).unsqueeze(1).to(config.device)  # (bsz, 1)
        init_behavior_user_feature = self.embedding(cls_user_token)  # bs * 1 * word_dim

        return init_feature, init_behavior_user_feature

    def forward(self, user_history, poi_history, input_pois, input_poi_words, ugc_label, rank_label, user_index,
                input_behavior_poi_index, input_behavior_labels, cl_user_index, model_name, is_train=True):

        # add_behavior, only_use_user_history = model_type
        if model_name == 'POT':
            model_type = (True, False)
        elif model_name == 'POT_woBehavior':
            model_type = (False, False)
        elif model_name == 'POT_woHGAT':
            model_type = (True, True)
        else:
            print("ERROR! Check Model Name.")

        if is_train:
            return self.train_one_batch((user_history, poi_history, input_pois, input_poi_words, ugc_label, rank_label,
                                         user_index, input_behavior_poi_index, input_behavior_labels), model_type)
        else:
            return self.evaluate_one_batch((user_history, poi_history, input_pois, input_poi_words, user_index,
                                            input_behavior_poi_index, input_behavior_labels), model_type)

    def train_one_batch(self, pairs_data, model_type):

        add_behavior, only_use_user_history = model_type

        user_history, poi_history, input_pois, input_poi_words, ugc_label, rank_label, user_index, \
        input_behavior_poi_index, input_behavior_labels = pairs_data

        word_dim, _, _, user_dim = self.feature_dim
        bs = user_history.size(0)

        # ========= Graph Encoder =============
        user_history_feature = self.embedding(user_history)
        poi_history_feature = self.embedding(poi_history)

        user_history_mask = user_history.data.eq(config.PAD_idx).to(config.device)
        user_aspect_self_mask = torch.ones(bs, user_history.size(1), user_history.size(2), user_history.size(3),
                                           1).data.eq(config.PAD_idx).to(config.device)  # self-aspect
        user_history_mask = torch.cat((user_history_mask, user_aspect_self_mask), dim=-1)
        # bs * max_up+1 * 1+neg * max_ua * max_uaw+1

        poi_history_mask = poi_history.data.eq(config.PAD_idx).to(config.device)
        poi_aspect_self_mask = torch.ones(bs, poi_history.size(1), poi_history.size(2),
                                          1).data.eq(config.PAD_idx).to(config.device)  # self-aspect
        poi_history_mask = torch.cat((poi_history_mask, poi_aspect_self_mask), dim=-1)
        # bs * max_up+1 * max_pa * max_paw+1

        input_user_poi_mask = input_pois[:, :-1].data.eq(config.PAD_idx).to(config.device)
        input_user_self_mask = torch.ones(bs, 1).data.eq(config.PAD_idx).to(config.device)  # self-user
        input_user_poi_mask = torch.cat((input_user_poi_mask, input_user_self_mask), dim=-1)

        inputs_feature = (user_history_feature, poi_history_feature)
        inputs_mask = (user_history_mask, poi_history_mask, input_user_poi_mask)
        prior_feature, init_behavior_user_feature = self.get_init_feature(bs)

        user_review_feature, poi_feature, cl_feature = self.graph_encoder(inputs_feature, inputs_mask, prior_feature,
                                                                          only_use_user_history)
        cl_loss = self.cl_criterion(cl_feature)

        # ======== Transformer Encoder =========
        input_poi_words_embed = self.embedding(input_poi_words)
        mask_src = input_poi_words.data.eq(config.PAD_idx).unsqueeze(1).to(config.device)
        # pad的地方全为True (需要mask的地方为true) bs * 1 * word_number
        input_poi_words_feature = self.trans_encoder(input_poi_words_embed, mask_src)  # bs * word_number * word_dim

        # ======== Behavior ==========
        behavior_matrix = ((self.cilck_sparse_graph, self.favor_sparse_graph, self.consume_sparse_graph),
                           self.uid_embed, self.pid_embed)
        behavior_data = (user_index, input_behavior_poi_index, input_behavior_labels)
        select_loss, user_behavior_feature = self.behavior_module(behavior_matrix, behavior_data)

        # ======== Decoder - Aspect ============
        aspect_bos_feature = self.bos_linear(poi_feature)  # bs * poi_dim

        if add_behavior:
            aspect_user_feature = torch.cat((user_review_feature, user_behavior_feature), dim=-1)  # bs * user_dim*2
            aspect_user_feature = self.user_feature_linear(aspect_user_feature)  # bs * user_dim
        else:
            aspect_user_feature = user_review_feature

        aspect_decoder_input = torch.cat((aspect_bos_feature, aspect_user_feature), dim=-1).unsqueeze(1)
        aspect_decoder_input = self.decoder_input_linear(aspect_decoder_input)  # bs * 1 * word_dim

        aspect_mask_trg = torch.ones(bs, 1, 1, dtype=torch.long).data.eq(config.PAD_idx).to(config.device)
        aspect_pre_logit, _ = self.decoder(aspect_decoder_input, input_poi_words_feature, (mask_src, aspect_mask_trg))
        aspect_logit = self.generator(aspect_pre_logit, is_generate_aspect=True,
                                      batch_output_word=None, is_greedy=False)  # bs * aspect
        rank_loss = get_rank_loss(aspect_logit, rank_label) * 0.75

        # ======== Decoder - Tags ==============
        _, top_number, max_label_length = ugc_label.size()
        tags_label = ugc_label.reshape(-1, max_label_length)  # bs*top_number * label_length

        word_poi_feature = self.bos_linear(poi_feature)
        word_bos_feature = word_poi_feature.unsqueeze(dim=1).repeat([1, top_number, 1])
        word_bos_feature = word_bos_feature.reshape(-1, word_dim).unsqueeze(1)  # bs*t * 1 * word_dim
        ugc_embed = self.embedding(tags_label)
        input_ugc = torch.cat((word_bos_feature, ugc_embed), dim=1)[:, :-1, :]  # bs*t * label_length * word_dim

        if add_behavior:
            word_user_feature = torch.cat((user_review_feature, user_behavior_feature), dim=-1)  # bs * user_dim*2
            word_user_feature = self.user_feature_linear(word_user_feature)  # bs * user_dim
        else:
            word_user_feature = user_review_feature

        word_user_feature = word_user_feature.unsqueeze(dim=1).unsqueeze(dim=1).repeat([1, top_number, max_label_length, 1])
        word_user_feature = word_user_feature.reshape(-1, max_label_length, user_dim)
        word_decoder_input = torch.cat((input_ugc, word_user_feature), dim=-1)
        word_decoder_input = self.decoder_input_linear(word_decoder_input)  # bs * label_length * word_dim
        # decoder_input => bs*t * label_length * word_dim

        word_mask_trg = tags_label.data.eq(config.PAD_idx).unsqueeze(1).to(config.device)

        word_input_poi_words_feature = input_poi_words_feature.unsqueeze(1).repeat([1, top_number, 1, 1])
        word_input_poi_words_feature = word_input_poi_words_feature.reshape(-1, word_input_poi_words_feature.size(-2),
                                                                            word_input_poi_words_feature.size(-1))
        # bs * 1 * word_number => bs*t * 1 * word_number
        word_mask_src = mask_src.unsqueeze(1).repeat([1, top_number, 1, 1]).reshape(-1, 1, mask_src.size(-1))
        word_pre_logit, _ = self.decoder(word_decoder_input, word_input_poi_words_feature, (word_mask_src, word_mask_trg))

        word_logit = self.generator(word_pre_logit, is_generate_aspect=False, batch_output_word=None, is_greedy=False)
        real_word_logit = word_logit[:, 1:, :]
        real_tags_label = tags_label[:, 1:]
        generation_loss = self.generate_criterion(real_word_logit.contiguous().view(-1, real_word_logit.size(-1)),
                                                  real_tags_label.contiguous().view(-1))

        # ================ Loss ====================
        if add_behavior:
            batch_loss = generation_loss + rank_loss + cl_loss + select_loss
            loss_set = torch.tensor([generation_loss.item(), rank_loss.item(), cl_loss.item(),
                                     select_loss.item()], dtype=torch.float).to(config.device).unsqueeze(dim=0)
        else:
            batch_loss = generation_loss + rank_loss + cl_loss
            loss_set = torch.tensor([generation_loss.item(), rank_loss.item(), cl_loss.item(),
                                     0], dtype=torch.float).to(config.device).unsqueeze(dim=0)

        return batch_loss, loss_set, cl_feature[1]

    def evaluate_one_batch(self, evaluate_pairs_data, model_type):

        add_behavior, only_use_user_history = model_type
        user_history, poi_history, input_pois, input_poi_words, user_index, \
        input_behavior_poi_index, input_behavior_labels = evaluate_pairs_data
        word_dim, _, _, user_dim = self.feature_dim
        bs = user_history.size(0)

        # ========= Graph Encoder =============
        user_history_feature = self.embedding(user_history)
        poi_history_feature = self.embedding(poi_history)

        user_history_mask = user_history.data.eq(config.PAD_idx).to(config.device)
        user_aspect_self_mask = torch.ones(bs, user_history.size(1), user_history.size(2), user_history.size(3),
                                           1).data.eq(config.PAD_idx).to(config.device)
        user_history_mask = torch.cat((user_history_mask, user_aspect_self_mask), dim=-1)
        # bs * max_up+1 * 1+neg * max_ua * max_uaw+1

        poi_history_mask = poi_history.data.eq(config.PAD_idx).to(config.device)
        poi_aspect_self_mask = torch.ones(bs, poi_history.size(1), poi_history.size(2),
                                          1).data.eq(config.PAD_idx).to(config.device)
        poi_history_mask = torch.cat((poi_history_mask, poi_aspect_self_mask), dim=-1)
        # bs * max_up+1 * max_pa * max_paw+1

        input_user_poi_mask = input_pois[:, :-1].data.eq(config.PAD_idx).to(config.device)
        input_user_self_mask = torch.ones(bs, 1).data.eq(config.PAD_idx).to(config.device)
        input_user_poi_mask = torch.cat((input_user_poi_mask, input_user_self_mask), dim=-1)

        inputs_feature = (user_history_feature, poi_history_feature)
        inputs_mask = (user_history_mask, poi_history_mask, input_user_poi_mask)
        prior_feature, init_behavior_user_feature = self.get_init_feature(bs)

        user_review_feature, poi_feature, cl_feature = self.graph_encoder(inputs_feature, inputs_mask, prior_feature,
                                                                          only_use_user_history)

        # ===== Trans Encoder ======
        input_poi_words_embed = self.embedding(input_poi_words)
        mask_src = input_poi_words.data.eq(config.PAD_idx).unsqueeze(1).to(config.device)
        input_poi_words_feature = self.trans_encoder(input_poi_words_embed, mask_src)  # bs * word_number * word_dim

        # ===== Behavior ======
        behavior_matrix = ((self.cilck_sparse_graph, self.favor_sparse_graph, self.consume_sparse_graph),
                           self.uid_embed, self.pid_embed)
        behavior_data = (user_index, input_behavior_poi_index, input_behavior_labels)
        _, user_behavior_feature = self.behavior_module(behavior_matrix, behavior_data)

        # ======== Decoder - Aspect ============
        aspect_bos_feature = self.bos_linear(poi_feature)  # bs * poi_dim

        if add_behavior:
            aspect_user_feature = torch.cat((user_review_feature, user_behavior_feature), dim=-1)  # bs * user_dim*2
            aspect_user_feature = self.user_feature_linear(aspect_user_feature)  # bs * user_dim
        else:
            aspect_user_feature = user_review_feature

        aspect_decoder_input = torch.cat((aspect_bos_feature, aspect_user_feature), dim=-1).unsqueeze(
            1)  # bs * word_dim+user_dim
        aspect_decoder_input = self.decoder_input_linear(aspect_decoder_input)  # bs * 1 * word_dim

        aspect_mask_trg = torch.ones(bs, 1, 1, dtype=torch.long).data.eq(config.PAD_idx).to(config.device)
        aspect_pre_logit, _ = self.decoder(aspect_decoder_input, input_poi_words_feature, (mask_src, aspect_mask_trg))
        aspect_logit = self.generator(aspect_pre_logit, is_generate_aspect=True,
                                      batch_output_word=None, is_greedy=True)  # bs * aspect
        _, generate_aspect_index = aspect_logit.data.topk(self.top_number, dim=-1)
        generate_aspect = generate_aspect_index + config.begin  # bs * top_number

        # ======== Decoder - Tags ==============
        word_poi_feature = self.bos_linear(poi_feature).unsqueeze(dim=1).repeat([1, self.top_number, 1])
        word_bos_feature = word_poi_feature.reshape(-1, word_dim).unsqueeze(1)  # bs*t * 1 * word_dim
        aspect_embed = self.embedding(generate_aspect).reshape(-1, word_dim).unsqueeze(1)  # bs*t * 1 * word_dim
        input_ugc = torch.cat((word_bos_feature, aspect_embed), dim=1)  # bs*t * 2 * word_dim

        if add_behavior:
            word_user_feature = torch.cat((user_review_feature, user_behavior_feature), dim=-1)  # bs * user_dim*2
            word_user_feature = self.user_feature_linear(word_user_feature)  # bs * user_dim
        else:
            word_user_feature = user_review_feature

        word_user_feature = word_user_feature.unsqueeze(dim=1).unsqueeze(dim=1).repeat([1, self.top_number, 2, 1])
        word_user_feature = word_user_feature.reshape(-1, 2, user_dim)
        word_decoder_input = torch.cat((input_ugc, word_user_feature), dim=-1)
        word_decoder_input = self.decoder_input_linear(word_decoder_input)  # bs*t * 2 * word_dim

        word_mask_trg = torch.ones(bs*self.top_number, 2, 1, dtype=torch.long).data.eq(config.PAD_idx).to(config.device)
        batch_output_word = torch.zeros(bs*self.top_number, 1, dtype=torch.long).to(config.device)  # bs*t * 1

        word_input_poi_words_feature = input_poi_words_feature.unsqueeze(1).repeat([1, self.top_number, 1, 1])
        word_input_poi_words_feature = word_input_poi_words_feature.reshape(-1, word_input_poi_words_feature.size(-2),
                                                                            word_input_poi_words_feature.size(-1))
        # bs * 1 * word_number => bs*t * 1 * word_number
        word_mask_src = mask_src.unsqueeze(1).repeat([1, self.top_number, 1, 1]).reshape(-1, 1, mask_src.size(-1))

        for step in range(self.max_label_length-1):

            pre_logit, _ = self.decoder(word_decoder_input, word_input_poi_words_feature, (word_mask_src, word_mask_trg))
            cul_logit = self.generator(pre_logit, is_generate_aspect=False,
                                       batch_output_word=batch_output_word, is_greedy=True)  # bs*t * vocab_size

            _, step_word_index = cul_logit.data.topk(1)  # bs*t * 1
            step_feature = self.embedding(step_word_index)

            step_feature = torch.cat((step_feature, word_user_feature[:, -1, :].unsqueeze(1)), dim=-1)
            step_feature = self.decoder_input_linear(step_feature)  # bs*t * 1 * word_dim

            word_decoder_input = torch.cat((word_decoder_input, step_feature), dim=1)
            # decoder_input:               bs*t * length * word_dim
            # embedding(step_word_index):  bs*t *    1   * word_dim
            word_mask_trg = torch.cat((word_mask_trg, step_word_index.data.eq(config.PAD_idx).unsqueeze(1)), dim=1)

            if step == 0:
                batch_output_word = step_word_index
            else:
                batch_output_word = torch.cat((batch_output_word, step_word_index), dim=-1)

        # batch_output_word bs*t * (len-1)
        output_aspect = generate_aspect.reshape(-1).unsqueeze(-1)  # bs*t * 1
        batch_output = torch.cat((output_aspect, batch_output_word), dim=-1)  # bs*t * len

        return batch_output  # bs*t * max_length

