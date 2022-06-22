
def get_ngrams(ngram_size, prev_input_ids, bs):
    generated_ngrams = [{} for _ in range(bs)]
    for idx in range(bs):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    # [bs], dict:{(A, B, C): D, (B, C, D): E}
    return generated_ngrams


def get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def get_banned_ngram_tokens(ngram_size, prev_input_ids, bs, cur_len):

    if cur_len + 1 < ngram_size:  # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(bs)]
    else:
        generated_ngrams = get_ngrams(ngram_size, prev_input_ids, bs)

        banned_tokens = [
            get_generated_ngrams(generated_ngrams[bs_idx], prev_input_ids[bs_idx], ngram_size, cur_len)
            for bs_idx in range(bs)
        ]
        # bs * 1 [[],[],[],[]]
        return banned_tokens


class NoRepeatNGramLogitsProcessor():

    def __init__(self, ngram_size):
        self.ngram_size = ngram_size

    def get_scores(self, input_ids, scores):  # scores => bs * vocab_size
        bs = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = get_banned_ngram_tokens(self.ngram_size, input_ids, bs, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores