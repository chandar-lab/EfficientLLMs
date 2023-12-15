# code modified from: https://github.com/cybertronai/bflm/blob/master/eval_lambada_slow.py

from evaluate import Evaluate, compute_perplexity
from tqdm import tqdm
from datasets import load_dataset
from sacremoses import MosesTokenizer, MosesDetokenizer
import torch

def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    return '\n' + text.strip()


@Evaluate.register("lambada")
class LAMBADA(Evaluate):
    def __init__(self, detokenize: bool = False, stop_word_filter: bool = False, stride: int = 512,
                 detokenize_havent: bool = False, task: str = 'accuracy', **kwargs):
        super().__init__(**kwargs)
        self.detokenize = detokenize
        self.stop_word_filter = stop_word_filter
        self.detokenize_havent = detokenize_havent
        self.task = task
        # dataset_name = "lambada" --> not the same dataset as openai used!
        dataset_name = "craffel/openai_lambada"
        self.raw_datasets = load_dataset(dataset_name, num_proc=4, split='test')
        self.stride = stride

    def compute(self, model, tokenizer):
        if self.task == 'accuracy':
            model.config.pad_token_id = model.config.eos_token_id  # to avoid warning

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
            bad_words_ids = []
            for word in stopwords:
                bad_words_ids.append(tokenizer(word).input_ids)

            correct = 0
            for i in tqdm(range(len(self.raw_datasets))):
                sample_text = self.raw_datasets[i]['text']
                if self.detokenize:
                    # detokenizer = MosesDetokenizer(lang='en')
                    # sample_text = detokenizer.detokenize(sample_text.split())
                    sample_text = preprocess(sample_text)
                if self.detokenize_havent:
                    sample_text.replace(" n't", "n't")
                prompt, label = sample_text.rsplit(' ', 1)

                inputs_label = tokenizer(label, return_tensors="pt").to(model.device)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                max_new_tokens = inputs_label.input_ids.shape[-1]

                if self.stop_word_filter:
                    output = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                                            max_new_tokens=max_new_tokens, num_beams=self.num_beams, top_k=self.top_k,
                                            top_p=self.top_p, do_sample=self.do_sample,
                                            bad_words_ids=bad_words_ids)
                else:
                    output = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                                            max_new_tokens=max_new_tokens, num_beams=self.num_beams, top_k=self.top_k,
                                            top_p=self.top_p, do_sample=self.do_sample, )

                predict = tokenizer.batch_decode(output, skip_special_tokens=True)[0].rsplit(' ', 1)[1]
                if label in predict:
                    correct += 1

            return correct / len(self.raw_datasets)

        elif self.task=='ppl':
            return compute_perplexity(model, tokenizer, self.raw_datasets, self.stride)

        else:
            raise NotImplementedError
