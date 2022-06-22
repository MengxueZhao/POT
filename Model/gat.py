import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter

from Utils import config
from Utils.embed import get_word_position_embedding


class GAT_Layer(nn.Module):
    def __init__(self, num_heads, input_dim, output_dim, attn_dropout=0.0):
        super(GAT_Layer, self).__init__()
        self.num_heads = num_heads

        self.self_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.w_head = Parameter(torch.Tensor(num_heads, output_dim, output_dim).to(config.device))
        self.a_src = Parameter(torch.Tensor(num_heads, output_dim, 1).to(config.device))
        self.a_dst = Parameter(torch.Tensor(num_heads, output_dim, 1).to(config.device))
        self.bias = Parameter(torch.Tensor(output_dim).to(config.device))

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        init.xavier_uniform_(self.w_head)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)
        init.constant_(self.bias, 0)

    def forward(self, x, prior_feature, x_mask):

        if prior_feature.size(-1) == 200:  # prior: word_dim
            x = torch.cat((x, prior_feature.unsqueeze(dim=-2)), dim=-2)
            h_prime = self.self_linear(x)
            # bs * poi_number+1 * user_number*2 * aspect_number * word_number+1 * aspect_dim
        else:
            x_linear = self.self_linear(x)
            h_prime = torch.cat((x_linear, prior_feature.unsqueeze(dim=-2)), dim=-2)  # self

        h_prime = torch.matmul(h_prime.unsqueeze(-3), self.w_head)

        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)

        x_mask = x_mask.unsqueeze(dim=-1)  # bs * max_up+1 * pos+neg * max_ua * max_uaw+1 * 1
        number = h_prime.size(-2)

        if len(h_prime.size()) == 7:  # cl-user_history: node_aspect_word
            attn = attn_src.expand(-1, -1, -1, -1, -1, -1, number) + \
                   attn_dst.expand(-1, -1, -1, -1, -1, -1, number).permute(0, 1, 2, 3, 4, 6, 5)
            # bs * poi_number+1 * user_number*2 * aspect_number * head_number * word_number+1 * word_number+1
            mask = x_mask + x_mask.permute(0, 1, 2, 3, 5, 4)

        elif len(h_prime.size()) == 6:  # node_aspect_word: poi_feature
            attn = attn_src.expand(-1, -1, -1, -1, -1, number) + \
                   attn_dst.expand(-1, -1, -1, -1, -1, number).permute(0, 1, 2, 3, 5, 4)
            # bs * poi_number+1 * aspect_number * head_number * word_number+1 * word_number+1
            mask = x_mask + x_mask.permute(0, 1, 2, 4, 3)

        elif len(h_prime.size()) == 4:  # p2u: user_feature
            attn = attn_src.expand(-1, -1, -1, number) + \
                   attn_dst.expand(-1, -1, -1, number).permute(0, 1, 3, 2)
            # bs * head_number * poi_number+1 * poi_number+1
            mask = x_mask + x_mask.permute(0, 2, 1)
        else:
            print("ERROR! Check GAT_Layer input.")

        mask = mask.unsqueeze(dim=-3)
        attn = self.leaky_relu(attn)
        # bs * poi_number+1 * user_number*2 * aspect_number * head_number * word_number+1 * word_number+1
        attn.data.masked_fill_(mask, -1e18)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.matmul(attn, h_prime) + self.bias
        output = output.mean(dim=-3)

        return output


class GAT_Graph(nn.Module):
    def __init__(self, num_heads, feature_dim, graph_data_range):
        super(GAT_Graph, self).__init__()

        word_dim, aspect_dim, poi_dim, user_dim = feature_dim
        max_ua_length, max_pa_length, max_up_length = graph_data_range

        self.gat_user_history = GAT_Layer(num_heads, word_dim, aspect_dim)
        self.aspect_user = nn.Linear(max_ua_length * aspect_dim, user_dim, bias=False)

        self.gat_poi_history = GAT_Layer(num_heads, word_dim, aspect_dim)
        self.aspect_poi = nn.Linear(max_pa_length * aspect_dim, poi_dim, bias=False)

        self.gat_pu = GAT_Layer(num_heads, poi_dim, user_dim)

        self.time_position = get_word_position_embedding(max_up_length, poi_dim)  # 正弦波信号

    def forward(self, inputs_feature, prior_feature, inputs_mask, only_use_user_history):

        user_history_words_feature, poi_history_words_feature = inputs_feature
        prior_user_aspect_feature, prior_poi_aspect_feature, prior_user_feature = prior_feature
        user_history_words_mask, poi_history_words_mask, input_user_poi_mask = inputs_mask

        # ==user_history_feature==
        user_history_aspect_output = self.gat_user_history(user_history_words_feature, prior_user_aspect_feature,
                                                            user_history_words_mask)
        # bs * poi_number+1 * user_number*2+1 * aspect_number * word_number+1 * aspect_dim
        user_history_aspect_feature = user_history_aspect_output[:, :, :, :, -1, :]

        user_history_aspect_merge_feature = user_history_aspect_feature.reshape(user_history_aspect_feature.size(0),
                                                                                user_history_aspect_feature.size(1),
                                                                                user_history_aspect_feature.size(2), -1)
        # print(user_history_aspect_merge_feature.shape)
        user_history_feature = self.aspect_user(user_history_aspect_merge_feature)
        # bs * poi_number+1 * user_number*2+1 * user_dim

        # ==poi_history_feature==
        # poi_history_words_feature # bs * max_up+1 * max_pa * max_paw * word_dim
        poi_history_aspect_output = self.gat_poi_history(poi_history_words_feature, prior_poi_aspect_feature,
                                                          poi_history_words_mask)
        # bs * poi_number+1 * aspect_number * word_number+1 * aspect_dim
        poi_history_aspect_feature = poi_history_aspect_output[:, :, :, -1, :]
        poi_history_aspect_merge_feature = poi_history_aspect_feature.reshape(poi_history_aspect_feature.size(0),
                                                                              poi_history_aspect_feature.size(1),
                                                                              -1)
        poi_history_feature = self.aspect_poi(poi_history_aspect_merge_feature)
        # bs * poi_number+1 * poi_dim

        # CL
        cl_feature = (poi_history_feature, user_history_feature)

        if only_use_user_history:
            input_user_feature = user_history_feature[:, -1, 0, :]

        else:
            if prior_user_feature is None:
                prior_user_feature = user_history_feature[:, -1, 0, :]

            input_user_poi_feature = poi_history_feature[:, :-1, :]  # bs * max_up_length * poi_dim

            input_user_poi_feature += self.time_position[:, :input_user_poi_feature.size(1), :].type_as(
                input_user_poi_feature.data)
            # bs * max_up_length * poi_dim
            input_user_output = self.gat_pu(input_user_poi_feature, prior_user_feature, input_user_poi_mask)
            input_user_feature = input_user_output[:, -1, :]  # bs * user_dim

        self_feature = (user_history_aspect_feature, poi_history_aspect_feature, input_user_feature)

        return self_feature, cl_feature


