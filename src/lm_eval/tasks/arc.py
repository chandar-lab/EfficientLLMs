from .common import HFTask, simple_accuracy_metric
from tqdm import auto as tqdm_lib
import numpy as np


@HFTask.register("arc_easy")
class ARCEasy(HFTask):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Easy"
    TASK_NAME = 'arc_easy'

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc, include_target=True):
        q = "Question: " + doc['question'] + '\n'
        a = "Answer:" + (
            (" " + doc['choices']['text'][doc['choices']['label'].index(doc['answerKey'])]) if include_target else "")
        return q + a

    def evaluate(self, docs, lm, tokenizer):
        answerKey_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
                            '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, }
        golds = [answerKey_to_num[doc["answerKey"]] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
            )
            probs = []
            for choice in doc['choices']['text']:
                probs.append(lm.loglikelihood(ctx, " " + self.convert_choice(choice), tokenizer=tokenizer))
            probs = np.array(probs)
            preds.append(np.argmax(probs))

        return simple_accuracy_metric(preds=preds, golds=golds)

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]


@HFTask.register("arc_challenge")
class ARCChallenge(ARCEasy):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Challenge"
    TASK_NAME = 'arc_challenge'
