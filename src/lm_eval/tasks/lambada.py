from tqdm import auto as tqdm_lib
from .common import HFTask
import datasets

@HFTask.register("lambada")
class Lambada(HFTask):
    DATASET_PATH = "craffel/openai_lambada"
    DATASET_NAME = None
    TASK_NAME = 'lambada'

    stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during',
                 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours',
                 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from',
                 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his',
                 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at',
                 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
                 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he',
                 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after',
                 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
                 'further', 'was', 'here', 'than'}

    def __init__(self, detokenize: bool = True, detokenize_havent: bool = True,
                 stop_word_filter: bool = False, num_beams: int = 5, top_k: int = 1, top_p: float = 0.,
                 do_sample: bool = False, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._training_docs = None
        self.data = datasets.load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        self.data["validation"] = self.data.pop('test')  # rename the test split to val
        self.detokenize = detokenize
        self.detokenize_havent = detokenize_havent
        self.stop_word_filter = stop_word_filter
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.temperature = temperature

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc, include_target=True):
        sample_text = self.preprocess(doc['text'])
        prompt, label = sample_text.rsplit(' ', 1)
        return prompt, label

    def evaluate(self, docs, lm, tokenizer):

        bad_words_ids = []
        for word in self.stopwords:
            bad_words_ids.append(tokenizer(word).input_ids)

        correct = 0
        for doc in tqdm_lib.tqdm(docs):
            prompt, label = self.doc_to_text(doc)

            inputs_label = tokenizer(label, return_tensors="pt").to(lm.device)
            inputs = tokenizer(prompt, return_tensors="pt").to(lm.device)
            max_new_tokens = inputs_label.input_ids.shape[-1]
            max_length = inputs.input_ids.shape[1] + max_new_tokens

            if self.stop_word_filter:
                output = lm.generate(inputs.input_ids, max_length=max_length, top_k=self.top_k,
                                     top_p=self.top_p, bad_words_ids=bad_words_ids)
            else:
                output = lm.generate(inputs.input_ids, max_length=max_length,
                                     top_k=self.top_k, top_p=self.top_p)

            predict = tokenizer.batch_decode(output, skip_special_tokens=True)[0].rsplit(' ', 1)[1]
            if label in predict:
                correct += 1

        acc = correct / len(self.raw_datasets)
        return {
            "major": acc,
            "minor": {"acc": acc},
            "higher_is_better": True,
        }

    def preprocess(self, text):
        if self.detokenize:
            text = text.replace("“", '"')
            text = text.replace("”", '"')
            text = text.replace("''", '"')
            text = text.replace("``", '"')
            text = '\n' + text.strip()
        if self.detokenize_havent:
            text.replace(" n't", "n't")
        return text

