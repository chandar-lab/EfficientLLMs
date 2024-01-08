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
    def __init__(self, task: str = 'accuracy', num_beams: int = 5, top_k: int = 1, top_p: float = 0.,
                 do_sample: bool = False, temperature: float = 1.0, detokenize: bool = False,
                 stop_word_filter: bool = False, stride: int = 512,
                 detokenize_havent: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.temperature = temperature
        self.detokenize = detokenize
        self.stop_word_filter = stop_word_filter
        self.detokenize_havent = detokenize_havent
        self.task = task
        # dataset_name = "lambada" --> not the same dataset as openai used!
        dataset_name = "craffel/openai_lambada"
        self.raw_datasets = load_dataset(dataset_name, num_proc=4, split='test')
        self.stride = stride

    def compute(self, model, tokenizer):
        model = model.eval()
        # cumsum_cuda_kernel in flash attention does not support deterministic operation.
        torch.use_deterministic_algorithms(False)
        device = torch.device('cuda')
        if self.task == 'accuracy':
            # model.config.pad_token_id = model.config.eos_token_id  # to avoid warning

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

                inputs_label = tokenizer(label, return_tensors="pt").to(device)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                max_new_tokens = inputs_label.input_ids.shape[-1]
                max_length = inputs.input_ids.shape[1] + max_new_tokens

                if self.stop_word_filter:
                    # output = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                    #                         max_new_tokens=max_new_tokens, num_beams=self.num_beams, top_k=self.top_k,
                    #                         top_p=self.top_p, do_sample=self.do_sample,
                    #                         bad_words_ids=bad_words_ids)
                    output = model.generate(inputs.input_ids, max_length=max_length, top_k=self.top_k,
                                            top_p=self.top_p, bad_words_ids=bad_words_ids)
                else:
                    # output = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                    #                         max_new_tokens=max_new_tokens, num_beams=self.num_beams, top_k=self.top_k,
                    #                         top_p=self.top_p, do_sample=self.do_sample, )
                    output = model.generate(inputs.input_ids, max_length=max_length,
                                            top_k=self.top_k, top_p=self.top_p)

                predict = tokenizer.batch_decode(output, skip_special_tokens=True)[0].rsplit(' ', 1)[1]
                if label in predict:
                    correct += 1

            return correct / len(self.raw_datasets)

        elif self.task=='ppl':
            return compute_perplexity(model, tokenizer, self.raw_datasets, self.stride)

        else:
            raise NotImplementedError


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from transformers import GPT2LMHeadModel
    from evaluate import Evaluate
    from common import Params
    import torch

    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": "main",
        "token": None,
        "trust_remote_code": False,
        "add_prefix_space": True
    }
    tokenizer = AutoTokenizer.from_pretrained("gpt2", **tokenizer_kwargs)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dtype=torch.bfloat16, device='cuda').eval()

    task = Evaluate.from_params(
        Params(
            {
                "type": "lambada",
                "task": 'accuracy',
                "num_beams": 5,
                "top_k": 100,
                "top_p": 0.9,
                "do_sample": True,
                "temperature": 1.0,
                "detokenize": True,
                "stop_word_filter": True,
                "detokenize_havent": False,

            }
        )
    )
    print("Accuracy:")
    print(task.compute(model, tokenizer))

    task = Evaluate.from_params(
        Params(
            {
                "type": "lambada",
                "task": 'ppl',
                "stride": 512,
            }
        )
    )
    print("Perplexity:")
    print(task.compute(model, tokenizer))

