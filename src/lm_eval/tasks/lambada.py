from lm_eval.base import Dataset
from lm_eval.utils import sh
import json
import requests
from .common import HFTask
from evaluate import LAMBADA


@HFTask.register("lambada")
class Lambada(HFTask):
    DATASET_PATH = "Rowan/hellaswag"
    DATASET_NAME = None
    TASK_NAME = 'lambada'

    def download(self):
        pass

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def load_doc(self, myjson):
        pass

    def test_docs(self):
        pass

    def doc_to_text(self, doc, include_target=True):
        pass

    def evaluate(self, docs, lm, tokenizer):
        # task_ = LAMBADA(task='accuracy',
        #                 top_k=1,
        #                 top_p=0.,
        #                 temperature=1.0,
        #                 detokenize=True,
        #                 stop_word_filter=False,
        #                 detokenize_havent=False,
        #                 stride=1024, )
        #
        # print(task_)
        # acc = task_.compute(lm.gpt2, lm.tokenizer)
        # return {
        #     "major": acc,
        #     "minor": {"acc": acc},
        #     "higher_is_better": True,
        # }
        raise NotImplementedError
