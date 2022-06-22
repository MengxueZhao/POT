import numpy as np
import torch
from Utils import config


def get_ndcg_weight(predict, label):
    # rank_label: bs * aspect_number
    bs = predict.size(0)
    aspect_number = len(config.aspect_token)
    (sorted_label_scores, sorted_label_index) = label.sort(dim=-1, descending=True)

    IDCG_num = 2 ** sorted_label_scores - 1  # bs * aspect_number
    IDCG_den = torch.tensor(np.log2(np.arange(aspect_number) + 1 + 1), dtype=torch.float).to(config.device)
    IDCG_den = IDCG_den.unsqueeze(0).repeat([bs, 1])  # bs * aspect_number
    max_DCG = (IDCG_num / IDCG_den).sum(dim=-1)  # bs

    G_i = (2 ** label - 1) / (max_DCG.unsqueeze(-1).repeat([1, aspect_number]))  # bs * aspect
    G_i_ext = G_i.unsqueeze(-1).repeat([1, 1, aspect_number])
    G_diffs = G_i_ext - G_i_ext.permute(0, 2, 1)

    D_i = 1 / IDCG_den  # bs * aspect
    D_i_ext = D_i.unsqueeze(-1).repeat([1, 1, aspect_number])
    D_diffs = D_i_ext - D_i_ext.permute(0, 2, 1)

    delta_ndcg_diffs = G_diffs.abs() * D_diffs.abs()

    return delta_ndcg_diffs


def get_rank_mask(label, predict_diffs):
    # label: bs * aspect_number
    aspect_number = label.size(1)

    # yi > yj
    label = label.unsqueeze(-1).repeat([1, 1, aspect_number])  # bs * aspect_number * aspect_number
    label_mask = (label - label.permute(0, 2, 1)) > 0
    # label_mask = label_mask * triangle_mask

    # si < sj
    predict_mask = predict_diffs < 0

    rank_mask = label_mask * predict_mask
    return rank_mask


def get_rank_loss(logit, label):
    # logit: bs * aspect
    predict = logit.unsqueeze(-1).repeat([1, 1, logit.size(1)])
    predict_diffs = predict - predict.permute(0, 2, 1)

    delta = 1
    simple_loss = (1 + (-1 * delta * predict_diffs).exp()).log2()
    ndcg_weight = get_ndcg_weight(logit, label)
    rank_loss = simple_loss * get_rank_mask(label, predict_diffs) * ndcg_weight

    return rank_loss.sum() / rank_loss.size(0)





