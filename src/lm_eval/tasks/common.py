import datasets
import numpy as np
import random
import torch
from tqdm import auto as tqdm_lib
from rouge_score import rouge_scorer, scoring
import collections
import math
from common import Registrable
from ..base import Dataset


class HFTask(Dataset, Registrable):
    DATASET_PATH = None
    DATASET_NAME = None
    TASK_NAME = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._training_docs = None
        self.data = datasets.load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)

    def has_training_docs(self):
        """Whether the task has a training set"""
        return True if "train" in self.data.keys() else False

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return True if "validation" in self.data.keys() else False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True if "test" in self.data.keys() else False

    def training_docs(self):
        # Cache training for faster few-shot.
        # If data is too large to fit in memory, override this method.
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.data["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test"]

    def fewshot_examples(self, k):
        training_docs = self.training_docs()
        n = len(training_docs)
        indices = random.sample(range(n), k)
        return [training_docs[i] for i in indices]


def simple_accuracy_metric(preds, golds):
    acc = float((np.array(preds) == np.array(golds)).mean())
    return {
        "major": acc,
        "minor": {"acc": acc},
        "higher_is_better": True,
    }

def yesno(x):
    if x:
        return 'yes'
    else:
        return 'no'

# code from: https://huggingface.co/docs/transformers/perplexity
def compute_perplexity(model, tokenizer, raw_dataset, stride):
    key_name = next(iter(raw_dataset.features))
    encodings = tokenizer("\n\n".join(raw_dataset[key_name]), return_tensors="pt")
    max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm_lib.tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


# modified code from: https://github.com/huggingface/evaluate/blob/dfbbf15ae66a4eb1053be6bb09d8bfa6573f7304/metrics/rouge/rouge.py
def get_rouge_score(predictions, references, tokenizer):
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    multi_ref = isinstance(references[0], list)

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False, tokenizer=tokenizer)
    aggregator = scoring.BootstrapAggregator()

    for ref, pred in zip(references, predictions):
        if multi_ref:
            score = scorer.score_multi(ref, pred)
        else:
            score = scorer.score(ref, pred)
        aggregator.add_scores(score)

    result = aggregator.aggregate()
    for key in result:
        result[key] = result[key].mid.fmeasure

    return result

# code from: https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts

# code from: https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return bleu, precisions, bp, ratio, translation_length, reference_length
