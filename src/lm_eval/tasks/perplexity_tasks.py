from .common import HFTask, compute_perplexity
import datasets
import torch


@HFTask.register("wikitext_2")
class WikiText2(HFTask):
    DATASET_PATH = "wikitext"
    DATASET_NAME = 'wikitext-2-v1'
    TASK_NAME = 'wikitext-2'

    def __init__(self,  stride: int = 512, **kwargs):
        super().__init__(**kwargs)
        self._training_docs = None
        self.data = datasets.load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME, split='test')
        self.data["validation"] = self.data.pop('test')  # rename the test split to val
        self.stride = stride

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        return ""

    def doc_to_text(self, doc, include_target=True):
        pass

    @torch.inference_mode()
    def evaluate(self, docs, lm, tokenizer):
        ppl = compute_perplexity(lm, tokenizer, docs, self.stride)
        return {
            "major": ppl,
            "minor": {"perplexity": ppl},
            "higher_is_better": False,
        }


@HFTask.register("wikitext_103")
class WikiText103(WikiText2):
    DATASET_PATH = "wikitext"
    DATASET_NAME = 'wikitext-103-v1'
    TASK_NAME = 'wikitext_103'


@HFTask.register("ptb")
class PTB(WikiText2):
    DATASET_PATH = "ptb_text_only"
    DATASET_NAME = None
    TASK_NAME = 'ptb'


@HFTask.register("1bw")
class OneBillionWord(WikiText2):
    DATASET_PATH = "lm1b"
    DATASET_NAME = None
    TASK_NAME = '1bw'

