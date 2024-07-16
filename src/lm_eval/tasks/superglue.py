import numpy as np
from tqdm import auto as tqdm_lib
from . common import HFTask, simple_accuracy_metric, yesno


@HFTask.register("boolq")
class BoolQ(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "boolq"
    TASK_NAME = 'boolq'

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "Read the following passages and answer each question with a yes or a no."

    def doc_to_text(self, doc, include_target=True):
        return f"{doc['passage']}\nquestion: {doc['question']}\nanswer: " \
            + (yesno(doc['label']) if include_target else "")

    def evaluate(self, docs, lm, tokenizer):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in docs:
            ctx = self.fewshot_context(
                doc=doc,
            )
            preds.append(lm.loglikelihood(ctx, ' yes', tokenizer=tokenizer) > lm.loglikelihood(ctx, ' no', tokenizer=tokenizer))
        return simple_accuracy_metric(preds=preds, golds=golds)


@HFTask.register("commitmentbank")
class CommitmentBank(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "cb"
    TASK_NAME = 'commitmentbank'

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "{}\nquestion:\t{}\ttrue, false or neither?\nanswer:".format(
            doc["premise"],
            doc["hypothesis"],
        )
        if include_target:
            # True = entailment
            # False = contradiction
            # Neither = neutral
            text += " {}".format({0: "true", 1: "neither", 2: "false"}[doc["label"]])
        return text

    def evaluate(self, docs, lm, tokenizer):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
            )
            probs = np.array([
                lm.loglikelihood(ctx, ' true', tokenizer=tokenizer),
                lm.loglikelihood(ctx, ' neither', tokenizer=tokenizer),
                lm.loglikelihood(ctx, ' false', tokenizer=tokenizer),
            ])
            preds.append(np.argmax(probs))
        return simple_accuracy_metric(preds=preds, golds=golds)


@HFTask.register("copa")
class Copa(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "copa"
    TASK_NAME = 'copa'

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        # Drop the period
        connector = {
            "cause": "because",
            "effect": "therefore",
        }[doc["question"]]
        text = doc["premise"].strip()[:-1] + f" {connector} "
        if include_target:
            correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
            # Connect the sentences
            text += self.convert_choice(correct_choice)
        return text

    def evaluate(self, docs, lm, tokenizer):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
            )
            choice1 = " " + self.convert_choice(doc["choice1"])
            choice2 = " " + self.convert_choice(doc["choice2"])
            preds.append(lm.loglikelihood(ctx, choice2, tokenizer=tokenizer) > lm.loglikelihood(ctx, choice1, tokenizer=tokenizer))
        return simple_accuracy_metric(preds=preds, golds=golds)

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]


@HFTask.register("multirc")
class MultiRC(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "multirc"
    TASK_NAME = 'multirc'

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "READING COMPREHENSION ANSWER KEY"

    def doc_to_text(self, doc, include_target=True):
        return f"{doc['paragraph']}\n\n{doc['question']}\n" \
            + (self.format_answer(answer=doc["answer"], label=doc["label"])
               if include_target else "")

    @staticmethod
    def format_answer(answer, label):
        label_str = "True" if label else "False"
        return f"[{label_str}] {answer}"

    def evaluate(self, docs, lm, tokenizer):
        preds = []
        for doc in docs:
            ctx = self.fewshot_context(
                doc=doc,
            )
            true_choice = self.format_answer(answer=doc["answer"], label=True)
            false_choice = self.format_answer(answer=doc["answer"], label=False)
            preds.append(
                lm.loglikelihood(ctx, f' {true_choice}', tokenizer=tokenizer)
                > lm.loglikelihood(ctx, f' {false_choice}', tokenizer=tokenizer)
            )

        # Only count as correct if all answers are labeled correctly for each question
        question_scoring_dict = {}
        for doc, pred in zip(docs, preds):
            question_id = doc["idx"]["question"]
            if question_id not in question_scoring_dict:
                question_scoring_dict[question_id] = []
            gold_label = doc["label"] == 1
            question_scoring_dict[question_id].append(gold_label == pred)
        acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
        return {
            "major": acc,
            "minor": {"acc": acc},
            "higher_is_better": True,
        }


@HFTask.register("wic")
class WordsInContext(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wic"
    TASK_NAME = 'wic'

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "{}\n{}\nquestion\tIs the word '{}' used in the same way in the" \
               " two sentences above?\nanswer:".format(
                    doc["sentence1"],
                    doc["sentence2"],
                    doc["sentence1"][doc["start1"]:doc["end1"]],
                )
        if include_target:
            text += " {}".format({0: "no", 1: "yes"}[doc["label"]])
        return text

    def evaluate(self, docs, lm, tokenizer):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
            )
            preds.append(lm.loglikelihood(ctx, ' yes', tokenizer=tokenizer) > lm.loglikelihood(ctx, ' no', tokenizer=tokenizer))
        return simple_accuracy_metric(preds=preds, golds=golds)


@HFTask.register("wsc")
class WinogradSchemaChallenge(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wsc"
    TASK_NAME = 'wsc'

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                # GPT-3 Paper's format only uses positive examples
                self._training_docs = [
                    doc for doc in
                    self.data["train"]
                    if doc["label"]
                ]
            return self._training_docs

    def fewshot_description(self):
        return "Final Exam with Answer Key\n" \
           "Instructions: Please carefully read the following passages. " \
           "For each passage, you must identify which noun the pronoun marked in *bold*" \
           " refers to.\n====="

    def doc_to_text(self, doc, include_target=True):
        raw_passage = doc["text"]
        passage = (
            raw_passage[:doc["span2_index"]]
            + "*{}*".format(doc["span2_text"])
            + raw_passage[doc["span2_index"] + len(doc["span2_text"]):]
        )
        pronoun = doc["span2_text"]
        text = (
            f"Passage: {passage}\n"
            + f"Question: In the passage above, what does the pronoun \"*{pronoun}*\" refer to?\n"
            + "Answer:"
        )
        if include_target:
            text += " {}".format(doc["span1_text"])
        return text

    def evaluate(self, docs, lm, tokenizer):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
            )
            to_predict = " " + doc["span1_text"]
            num_tokens = len(lm.tokenizer.tokenize(to_predict))
            generated = lm.generate(
                context=ctx,
                max_gen_length=num_tokens,
            )
            preds.append(1 if generated == to_predict else 0)
        return simple_accuracy_metric(preds=preds, golds=golds)
