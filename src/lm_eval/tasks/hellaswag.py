from .common import HFTask, simple_accuracy_metric
from tqdm import auto as tqdm_lib
import numpy as np
import re

@HFTask.register("hellaswag")
class HellaSwag(HFTask):
    DATASET_PATH = "Rowan/hellaswag"
    DATASET_NAME = None
    TASK_NAME = 'hellaswag'

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # TODO: figure out description
        # return "based on activity_label read the following sentence and complete the sentence\n"
        return ""

    def doc_to_text(self, doc, include_target=True):
        q = self.preprocess(doc["activity_label"] + ": " + doc['ctx'])
        a = " " + self.preprocess(
            (doc['endings'][int(doc['label'])]) if include_target else "")
        return q + a

    def evaluate(self, docs, lm, tokenizer):
        golds = [int(doc["label"]) for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
            )
            probs = []
            for endings in doc['endings']:
                probs.append(lm.loglikelihood(ctx, " " + self.preprocess(endings), tokenizer=tokenizer))
            probs = np.array(probs)
            preds.append(np.argmax(probs))

        return simple_accuracy_metric(preds=preds, golds=golds)

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]

    @staticmethod
    def preprocess(text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

