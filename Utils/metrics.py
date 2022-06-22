import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import csv
import os

from Utils import config


def get_init_metrics():

    metrics = {}
    keys = ['f1_3', 'f1_5', 'ndcg_3', 'ndcg_5', 'bleu_1', 'bleu_2', 'dist_1', 'dist_2']
    for key in keys:
        metrics[key] = 0.0
    return metrics


def get_distinct(output, n):
    i_index = list(range(0, len(output) - n))
    j_index = list(range(n, len(output)))
    assert len(i_index) == len(j_index)
    n_grams = [' '.join(output[i:j]) for i, j in zip(i_index, j_index)]

    if len(output) == 0 or len(n_grams) == 0:
        return 0.
    else:
        return len(n_grams) / len(output)


def get_ndcg_score(predict, label, top_k):
    (sorted_label_scores, sorted_label_index) = label.sort(dim=-1, descending=True)
    sorted_label_scores = sorted_label_scores[:top_k]

    IDCG_num = 2 ** sorted_label_scores - 1
    IDCG_den = torch.tensor(np.log2(np.arange(top_k) + 1 + 1), dtype=torch.float).to(config.device)
    max_DCG = (IDCG_num / IDCG_den).sum(dim=-1)
    
    (sorted_predict_scores, sorted_predict_index) = predict.sort(dim=-1, descending=True)
    sorted_predict_index = sorted_predict_index[:top_k]
    predict_score = torch.gather(label, dim=-1, index=sorted_predict_index)

    ndcg = (2 ** predict_score - 1) / IDCG_den
    ndcg_score = ndcg.sum(dim=-1) / max_DCG
    
    if sorted_predict_scores[0].item() == 0:
        return 0.0
    else:
        return ndcg_score.item()


def get_rank_f1_score(predict, label, top_k):

    label_score = label.bool()

    (sorted_predict_scores, sorted_predict_index) = predict.sort(dim=-1, descending=True)
    sorted_predict_index = sorted_predict_index[:top_k]
    predict_score = torch.zeros(len(label), dtype=torch.bool).to(config.device)

    predict_score[sorted_predict_index] = True

    match_score = label_score * predict_score

    label_number = torch.nonzero(label_score).size(0)
    predict_number = torch.nonzero(predict_score).size(0)
    match_number = torch.nonzero(match_score).size(0)

    precision = match_number * 1.0 / predict_number if predict_number > 0 else 0.0
    recall = match_number * 1.0 / label_number if label_number > 0 else 0.0
    f1_k = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if sorted_predict_scores[0].item() == 0:
        return 0.0
    else:
        return f1_k


def get_metrics(output, label, rank_output, label_rank):

    metrics = {}

    # ====== BLEU ======
    smooth = SmoothingFunction()
    bleu_1 = sentence_bleu([label], output, weights=(1.0, 0.0, 0, 0), smoothing_function=smooth.method1)
    bleu_2 = sentence_bleu([label], output, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
    metrics['bleu_1'] = bleu_1
    metrics['bleu_2'] = bleu_2

    # ====== Distinct =========
    metrics['dist_1'] = get_distinct(output, 1)
    metrics['dist_2'] = get_distinct(output, 2)

    # ========= NDCG ==========
    metrics['ndcg_3'] = get_ndcg_score(rank_output, label_rank, top_k=3)
    metrics['ndcg_5'] = get_ndcg_score(rank_output, label_rank, top_k=5)

    # ======== F1 =============
    metrics['f1_3'] = get_rank_f1_score(rank_output, label_rank, top_k=3)
    metrics['f1_5'] = get_rank_f1_score(rank_output, label_rank, top_k=5)

    return metrics


def add_lexical_diversity(metrics, tokenized, n_gram):
    for n in n_gram:
        n_grams_all = []
        for line in tokenized:
            n_grams = list(zip(*[line[i:] for i in range(n)]))
            n_grams_all += n_grams
        divers_name = 'divers_' + str(n)
        metrics[divers_name] = len(set(n_grams_all)) / len(n_grams_all)
    return metrics


def save_metrics(metrics_list, loss_list, is_do_train, save_path, model_name):

    csv_name = 'Metrics_' + str(model_name) + '.csv'
    ty = 'w' if is_do_train else 'a+'

    with open(os.path.join(save_path, csv_name), ty, encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for i, metrics in enumerate(metrics_list):
            print_metrics = {}
            for key in metrics:
                if metrics[key] < 1:
                    print_metrics[key] = int(metrics[key] * 10000) / 100.0
                else:
                    print_metrics[key] = int(metrics[key] * 100) / 100.0

            current_loss = loss_list[i]
            print_metrics['Lg'] = int(current_loss[0] * 10000) / 10000.0
            print_metrics['Lr'] = int(current_loss[1] * 10000) / 10000.0
            if len(current_loss) == 2:
                print_metrics['L'] = int((current_loss[0] + current_loss[1]) * 10000) / 10000.0
            elif len(current_loss) == 3:
                print_metrics['Lc'] = int(current_loss[2] * 10000) / 10000.0
                print_metrics['L'] = int((current_loss[0] + current_loss[1] + current_loss[2]) * 10000) / 10000.0
            elif len(current_loss) == 4:
                print_metrics['Lc'] = int(current_loss[2] * 10000) / 10000.0
                print_metrics['Ls'] = int(current_loss[3] * 10000) / 10000.0
                print_metrics['L'] = int((current_loss[0] + current_loss[1] +
                                          current_loss[2] + current_loss[3]) * 10000) / 10000.0
            else:
                print("ERROR! Check loss_list size.")

            write_list = []
            if i == 0:
                write_list.append("epoch")
                for key in print_metrics:
                    write_list.append(key)
                write_list.append("data")
                writer.writerow(write_list)

            if is_do_train:
                write_list = [i]
            else:
                write_list = ['-']
            for key in print_metrics:
                write_list.append(print_metrics[key])
            if is_do_train:
                write_list.append("dev")
            else:
                write_list.append("test")

            writer.writerow(write_list)


def get_tag_sequence(tag_tensor, index, vocab, sequence, print_sequence, aspect_output, rank_score, data_tags):

    for i in range(len(tag_tensor[index])):
        word_index = tag_tensor[index][i]
        if i == 0 and word_index.item() == config.EOS_idx:
            break

        if word_index.item() == config.PAD_idx:
            break

        if word_index.item() in config.aspect_token:  # aspect
            print_sequence.append(config.aspect_token2name[config.aspect_token[word_index.item()]])
            if aspect_output is not None:
                aspect_output[word_index.item() - config.begin] = rank_score  # aspect_index = rank_score
        else:  # not aspect
            sequence.append(vocab.index2word[word_index.item()])
            print_sequence.append(vocab.index2word[word_index.item()])
            data_tags.append(vocab.index2word[word_index.item()])

        if word_index.item() == config.EOS_idx:  # add EOS
            data_tags = data_tags[:-1]
            break

    return sequence, print_sequence, aspect_output, data_tags
