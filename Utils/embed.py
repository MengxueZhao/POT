import torch
import torch.nn as nn
import numpy as np
import math

from Utils import config


def gen_embeddings(vocab, word_dim, emb_dict):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, word_dim) * 0.01
    print('Embeddings: %d x %d' % (vocab.n_words, word_dim))
    pre_trained = len(emb_dict)

    for w in emb_dict:
        if len(emb_dict[w]) == word_dim:
            embeddings[vocab.word2index[w]] = emb_dict[w]
        else:
            print("ERROR! Word dim should be 200.")
    print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))

    return embeddings


class Embeddings(nn.Module):
    def __init__(self, vocab, word_dim, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, word_dim, padding_idx=padding_idx)
        self.word_dim = word_dim

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.word_dim)


def get_word_embedding(vocab, word_dim, emb_dict):

    embedding = Embeddings(vocab.n_words, word_dim, padding_idx=config.PAD_idx)

    pre_embedding = gen_embeddings(vocab, word_dim, emb_dict)
    embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
    embedding.lut.weight.data.requires_grad = True

    return embedding


def get_word_position_embedding(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
        Generates a [1, length, channels] timing signal consisting of sinusoids
        Adapted from:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def get_attn_subsequent_mask(size):  # mask:1
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (LongTensor):
        * subsequent_mask [1 x size x size]
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask.to(config.device)


def get_bias_mask(max_length):  # 1 => -inf
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)