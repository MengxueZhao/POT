import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from Utils import config
import Model.global_feature as glo


class GCN_Layer(nn.Module):
    def __init__(self, gcn_size):
        super(GCN_Layer, self).__init__()
        self.layer_number, embed_size = gcn_size
        self.final_user_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.final_poi_linear = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, up_behavior_graph, user_embed, poi_embed, batch_user):
        all_user_feature = user_embed.unsqueeze(0)  # 1 * user_number * embed
        all_poi_feature = poi_embed.unsqueeze(0)  # 1 * poi_number * embed
        layer_user_feature, layer_poi_feature = user_embed, poi_embed

        A = up_behavior_graph  # total_user_number * total_poi_number
        for _ in range(self.layer_number):
            layer_user_feature = torch.sparse.mm(A, layer_poi_feature)  # user_number * embed_size
            norm_user_feature = (layer_user_feature * ((1.0 / torch.norm(layer_user_feature, dim=1)).unsqueeze(1))).unsqueeze(0)
            all_user_feature = torch.cat((all_user_feature, norm_user_feature), dim=0)

            layer_poi_feature = torch.sparse.mm(A.t(), layer_user_feature)  # poi_number * embed_size
            norm_poi_feature = (layer_poi_feature * ((1.0 / torch.norm(layer_poi_feature, dim=1)).unsqueeze(1))).unsqueeze(0)
            all_poi_feature = torch.cat((all_poi_feature, norm_poi_feature), dim=0)
        user_feature = self.final_user_linear(all_user_feature.sum(dim=0))  # user_number * embed
        poi_feature = self.final_poi_linear(all_poi_feature.sum(dim=0))

        batch_user_feature = user_feature[batch_user]  # bs*25 * embed
        up_weight = torch.matmul(batch_user_feature, poi_feature.t())  # bs*25 * poi_number

        return up_weight, user_feature


class Select_Layer(nn.Module):
    def __init__(self, user_top_number, user_dim):
        super(Select_Layer, self).__init__()

        self.top_neighbor = user_top_number
        self.score_linear = nn.Linear(user_dim, user_dim, bias=True)  # 64 * 64

    def forward(self, current_user_index, id_user_feature):
        bs_id_feature = id_user_feature[current_user_index.squeeze(-1)].detach()  # bs * 64
        bs_feature = self.score_linear(bs_id_feature)  # bs * 64
        uu_score = torch.matmul(bs_feature, id_user_feature.t().detach())  # bs * total_user_number

        top_score, top_index = uu_score.data.topk(self.top_neighbor, dim=-1)  # bs * 15
        top_score = F.softmax(top_score, dim=-1)

        index = top_index.data.cpu()
        before_feature = glo.get_value('global_user_feature')
        neighbor_user_feature = before_feature[index]

        behavior_user_feature = torch.matmul(top_score.unsqueeze(dim=-2),
                                             neighbor_user_feature.to(top_score.device).detach()).squeeze(dim=1)
        del index, before_feature, neighbor_user_feature

        return behavior_user_feature


class Behavior_Module(nn.Module):
    def __init__(self, gcn_size, similar_neighbor, max_up_length, total_poi_number):
        super(Behavior_Module, self).__init__()
        self.w = Parameter(torch.Tensor(1, 3).to(config.device))
        init.xavier_uniform_(self.w)

        self.fw = Parameter(torch.Tensor(3, 1).to(config.device))
        init.xavier_uniform_(self.fw)

        gcn_layer_number, id_embed_size, \
        gat_head_number, gat_layer_number = [int(x) for x in gcn_size.strip().split(",")]

        gcn_para = (gcn_layer_number, id_embed_size)

        self.click_gcn_layer = GCN_Layer(gcn_para)
        self.favor_gcn_layer = GCN_Layer(gcn_para)
        self.consume_gcn_layer = GCN_Layer(gcn_para)
        self.k = max_up_length

        self.select_criterion = nn.BCEWithLogitsLoss()

        self.total_poi_number = total_poi_number
        self.select_layer = Select_Layer(similar_neighbor, id_embed_size)

    def forward(self, behavior_matrix, behavior_data):

        sparse_graphs, uid_embed, pid_embed = behavior_matrix
        click_sparse_graph, favor_sparse_graph, consume_sparse_graph = sparse_graphs
        user_index, input_behavior_poi_index, input_behavior_labels = behavior_data

        batch_user = user_index.reshape(-1)  # bs*25
        input_behavior_poi_index = input_behavior_poi_index.reshape(batch_user.size(0), -1)  # bs*25 * 24
        input_behavior_labels = input_behavior_labels.reshape(batch_user.size(0), -1)  # bs*25 * 24

        w = F.softmax(self.w, dim=1)  # w = self.w / (self.w.sum() + 1e-8)
        fw = F.softmax(self.fw, dim=0)

        # 为了避免除0
        click_select_up, click_user_feature = self.click_gcn_layer(click_sparse_graph, uid_embed, pid_embed, batch_user)
        favor_select_up, favor_user_feature = self.favor_gcn_layer(favor_sparse_graph + 1e-18 * click_sparse_graph,
                                                                   uid_embed, pid_embed, batch_user)
        consume_select_up, consume_user_feature = self.consume_gcn_layer(consume_sparse_graph + 1e-18 * click_sparse_graph,
                                                                         uid_embed, pid_embed, batch_user)

        alternative_graph = favor_select_up * w[0][0] + click_select_up * w[0][1] + consume_select_up * w[0][2]

        predict = torch.gather(alternative_graph, dim=1, index=input_behavior_poi_index)  # bs*25 * 24

        select_loss = self.select_criterion(predict, input_behavior_labels)  # bs*25 * 24

        index_tensor = torch.arange(0, user_index.size(0) * user_index.size(1),
                                    user_index.size(1)).long().to(config.device)

        current_user_index = batch_user[index_tensor]  # bs * 1

        gcn_user_feature = torch.cat(((torch.cat((favor_user_feature.unsqueeze(-1),
                                                  consume_user_feature.unsqueeze(-1)), dim=-1)),
                                      click_user_feature.unsqueeze(-1)), dim=-1)  # bs * 64 * 3
        behavior_user_id_feature = torch.matmul(gcn_user_feature, fw).squeeze(-1)  # bs * 64 * 1

        user_feature = self.select_layer(current_user_index, behavior_user_id_feature)

        return select_loss, user_feature

