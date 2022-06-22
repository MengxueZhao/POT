import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random
from torch.nn.utils import clip_grad_norm_
import pickle

from Utils import config
from Utils.data_split import construct_data
from Model.model import POT_Model
from Utils.metrics import get_init_metrics, get_metrics, save_metrics, get_tag_sequence
from Utils.metrics import add_lexical_diversity
import Model.global_feature as glo

args = config.parser_parameter()

device_ids = list(range(torch.cuda.device_count()))

n_gpu = len(device_ids)

evaluate_metrics = []
evaluate_loss = []

glo._init()

if args.do_train:
    glo.set_value('global_user_feature', torch.rand(70000, int(args.feature_dim.strip().split(",")[-1])) * 0.01)
else:
    with open(os.path.join(args.pkl_dir, 'best_memory.p'), "rb") as f:
        glo.set_value('global_user_feature', pickle.load(f))


def update_global_user_history_feature(user_history_feature, cl_user_index):
    # user_history_feature  # bs * poi_number+1 * 1+user_number * user_dim
    user_history_feature = user_history_feature.reshape(-1, user_history_feature.size(-1))
    cl_user_index = cl_user_index.reshape(-1)  # bs*(poi_number+1)*(1+user_number)

    before_feature = glo.get_value('global_user_feature')
    index = cl_user_index.data.cpu()
    value = user_history_feature.data.cpu()
    before_feature[index] = value
    glo.set_value('global_user_feature', before_feature)
    del index, value, before_feature


def evaluate(model, vocab, data_loader, pkl_file_name, is_print_output=False):

    if pkl_file_name is not None:
        checkpoint = torch.load(os.path.join(args.pkl_dir, pkl_file_name))
        del checkpoint['cilck_sparse_graph']
        del checkpoint['favor_sparse_graph']
        del checkpoint['consume_sparse_graph']
        model.load_state_dict(checkpoint, strict=False)

    metrics = get_init_metrics()
    total_generate_tags = []
    output_aspects = []
    data_number = 0

    model.eval()
    with torch.no_grad():
        loss = 0.0
        losses = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(config.device)
        if is_print_output:
            txt_file = open(os.path.join(args.pkl_dir, 'output_' + str(args.model_name) + '.txt'), 'w', encoding='utf-8')

        for _, pairs_data in enumerate(data_loader):

            if n_gpu == 1:
                pairs_data = [d.to(config.device) for d in pairs_data]

            # calculate loss
            batch_loss, loss_set, _ = model(*pairs_data, args.model_name, is_train=True)
            if n_gpu > 1:
                batch_loss = batch_loss.mean()
                loss_set = loss_set.mean(dim=0)
            else:
                loss_set = loss_set.squeeze(dim=0)

            loss += batch_loss.item()
            losses += loss_set

            # calculate metrics
            ugc_label = pairs_data[4]
            rank_label = pairs_data[5]

            _, top_number, max_label_length = ugc_label.size()

            batch_output = model(*pairs_data, args.model_name, is_train=False)  # bs*t * len
            tags_label = ugc_label.reshape(-1, max_label_length)  # bs*t * len

            output_sentence, print_output = [], []
            label_sentence, print_label = [], []
            data_tags = []
            rank_output = torch.zeros([len(config.aspect_token)], dtype=torch.float)
            for index in range(len(batch_output)):

                rank_score = top_number - (index % top_number)
                output_sentence, print_output, \
                rank_output, data_tags = get_tag_sequence(batch_output, index, vocab, output_sentence,
                                                          print_output, rank_output, rank_score, data_tags)
                label_sentence, print_label, \
                _, _ = get_tag_sequence(tags_label, index, vocab, label_sentence,
                                        print_label, None, rank_score, [])

                if (index + 1) % top_number == 0:
                    if not rank_label.is_cuda:
                        label_rank = rank_label[index // top_number].to(config.device)
                    else:
                        label_rank = rank_label[index//top_number]

                    data_metrics = get_metrics(output_sentence, label_sentence,
                                               rank_output.to(config.device), label_rank)

                    total_generate_tags.append(data_tags)
                    output_aspects.append(rank_output.data.tolist())

                    for key in metrics:
                        metrics[key] += data_metrics[key]
                    data_number += 1

                    if is_print_output:
                        print("output:", print_output)
                        print("label: ", print_label)
                        txt_file.write("output:{} \n".format(output_sentence))
                        txt_file.write("label:{} \n".format(label_sentence))

                    output_sentence, print_output = [], []
                    label_sentence, print_label = [], []
                    data_tags = []
                    rank_output = torch.zeros([len(config.aspect_token)], dtype=torch.float)

        loss = loss / len(data_loader)
        losses = losses / len(data_loader)
        evaluate_loss.append(losses)

        for key in metrics:
            metrics[key] = metrics[key] / data_number
        metrics = add_lexical_diversity(metrics, total_generate_tags, n_gram=[3, 5])
        evaluate_metrics.append(metrics)

        return loss, losses, metrics


def test(model, vocab, test_loader, pkl_file='best.pkl'):

    print('【 TESTING 】')
    print(f'Check the BEST pkl: [{pkl_file}]\'s performance on [test data].')

    _, _, metrics = evaluate(model, vocab, test_loader, pkl_file, is_print_output=True)

    print('The metrics are shown below.')
    for key in metrics:
        print('{:>10s}  {:.4f}'.format(key, metrics[key]))

    print("Test over.")


def validation(model, vocab, valid_loader, pkl_file, best_pkl_information):

    print('【 VALIDATION 】')
    print(f'Check the pkl: [{pkl_file}]\'s performance on [valid data].')

    pkl_loss, pkl_losses, pkl_metrics = evaluate(model, vocab, valid_loader, None)

    if len(pkl_losses) == 2:
        print('{:>10s}: {:>8.5f}  {:>2s}:{:>8.5f}  {:>2s}:{:>8.5f}'.format(
            "Valid_loss", pkl_loss, 'Lg', pkl_losses[0], 'Lr', pkl_losses[1]))
    elif len(pkl_losses) == 3:
        print('{:>10s}: {:>8.5f}  {:>2s}:{:>8.5f}  {:>2s}:{:>8.5f}  {:>2s}:{:>8.5f}'.format(
            "Valid_loss", pkl_loss, 'Lg', pkl_losses[0], 'Lr', pkl_losses[1], 'Lc', pkl_losses[2]))
    elif len(pkl_losses) == 4:
        print('{:>10s}: {:>8.5f}  {:>2s}:{:>8.5f}  {:>2s}:{:>8.5f}  {:>2s}:{:>8.5f}  {:>2s}:{:>8.5f}'.format(
            "Valid_loss", pkl_loss, 'Lg', pkl_losses[0], 'Lr', pkl_losses[1], 'Lc', pkl_losses[2], 'Ls', pkl_losses[3]))
    else:
        print("ERROR! Check Loss Set.")

    current_judge = pkl_metrics['f1_3'] + pkl_metrics['f1_5'] + pkl_metrics['ndcg_3'] + pkl_metrics['ndcg_5'] \
                    + pkl_metrics['bleu_1'] + pkl_metrics['bleu_2']

    if (best_pkl_information['loss'] - pkl_loss) < 0.2:
        if (best_pkl_information['loss'] - pkl_loss) > 0.1:
            best_pkl_information['loss'] = pkl_loss
            best_pkl_information['bad_pkl_case'] = 0
            if current_judge > best_pkl_information['judge']:
                best_pkl_information['pkl_file'] = pkl_file
                best_pkl_information['judge'] = current_judge

        elif (best_pkl_information['loss'] - pkl_loss) > 0:
            best_pkl_information['loss'] = pkl_loss
            best_pkl_information['bad_pkl_case'] += 1
            if current_judge > best_pkl_information['judge']:
                best_pkl_information['pkl_file'] = pkl_file
                best_pkl_information['judge'] = current_judge

        else:
            best_pkl_information['bad_pkl_case'] += 1
    else:
        best_pkl_information['loss'] = pkl_loss

    return best_pkl_information


def train(model, optimizer, dataset, best_pkl_info):
    '''
    train model + save checkpoint
    '''

    train_loader, valid_loader, test_loader = dataset

    print('【 TRAINING 】')
    start = time.perf_counter()
    # train_loss = 0
    model.train()
    optimizer.zero_grad()

    for epoch in tqdm(range(args.epoch_number)):
        epoch_loss = 0
        losses = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(config.device)

        for index, pairs_data in enumerate(tqdm(train_loader)):

            if n_gpu == 1:
                pairs_data = [d.to(config.device) for d in pairs_data]

            batch_loss, loss_set, update_feature = model(*pairs_data, args.model_name, is_train=True)

            update_global_user_history_feature(update_feature, pairs_data[-1])

            if n_gpu > 1:
                batch_loss = batch_loss.mean()
                loss_set = loss_set.mean(dim=0)
            else:
                loss_set = loss_set.squeeze(dim=0)

            epoch_loss += batch_loss.item()
            losses += loss_set

            batch_loss.backward()

            if n_gpu > 1:
                clip_grad_norm_(model.module.parameters(), max_norm=10, norm_type=2)
            else:
                clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

            optimizer.step()
            optimizer.zero_grad()

            if (index + 1) % args.print_seg == 0:
                print(" {:>5s}:{:>2d} {:>5s}:{:>6d} {:>5s}:{:>10.3f}  |  "
                      "{:>4s}: {:>8.5f}  {:>2s}:{:>8.5f}  {:>2s}:{:>8.5f}  {:>2s}:{:>8.5f}  {:>2s}:{:>8.5f}".format(
                    "Epoch", epoch, "Step", index + 1, "Time", time.perf_counter() - start,
                    'Loss', epoch_loss / (index + 1), 'Lg', losses[0] / (index + 1), 'Lr', losses[1] / (index + 1),
                    'Lc', losses[2] / (index + 1), 'Ls', losses[3] / (index + 1)))

        pkl_file = '{}.pkl'.format(epoch)
        if n_gpu > 1:
            torch.save(model.module.state_dict(), os.path.join(args.pkl_dir, pkl_file))
        else:
            torch.save(model.state_dict(), os.path.join(args.pkl_dir, pkl_file))

        with open(os.path.join(args.pkl_dir, '{}_memory.p'.format(epoch)), "wb") as f:
            epoch_feature = glo.get_value('global_user_feature')
            pickle.dump(epoch_feature, f)
            del epoch_feature

        best_pkl_info = validation(model, vocab, valid_loader, pkl_file, best_pkl_info)
        model.train()

        if best_pkl_info['bad_pkl_case'] > 1:
            break

    os.rename(os.path.join(args.pkl_dir, best_pkl_info['pkl_file']), os.path.join(args.pkl_dir, "best.pkl"))
    os.rename(os.path.join(args.pkl_dir, '{}_memory.p'.format(best_pkl_info['pkl_file'].split('.')[0])),
              os.path.join(args.pkl_dir, "best_memory.p"))
    print("Training over. The best pkl is {}, saved as \"best.pkl\", "
          "please use single GPU to test it.".format(best_pkl_info['pkl_file']))


if __name__ == '__main__':

    from torch.backends import cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    vocab, vector_dict, dataset, mini_range, behavior_matrix = construct_data(args, n_gpu)

    train_loader, valid_loader, test_loader = dataset

    model = POT_Model(args, vocab, vector_dict, mini_range, behavior_matrix)
    model = model.to(config.device)

    params = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    best_pkl_info = {'pkl_file': ' ', 'loss': 100, 'judge': 0, 'bad_pkl_case': 0}

    if args.do_train:
        train(model, optimizer, dataset, best_pkl_info)
    else:
        test(model, vocab, test_loader, "best.pkl")  # use single gpu to test

    save_metrics(evaluate_metrics, evaluate_loss, args.do_train, args.pkl_dir, args.model_name)

