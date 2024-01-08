from evaluate import Evaluate
from tqdm import tqdm
from datasets import load_dataset
from tabulate import tabulate
import warnings
import torch


def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    return '\n' + text.strip()


def show_result(options, options_log_prob, ids_options_list, use_scale=True, title='log prob'):
    sort_idx = options_log_prob.argsort(descending=True)
    min_scale = options_log_prob.exp().max().log10().round().item()
    probs = (options_log_prob.exp() * (10 ** (-min_scale))).detach().to('cpu').numpy().round(3)
    scores = []
    for i in range(len(options)):
        if use_scale:
            if ids_options_list:
                scores.append([options[sort_idx[i]], probs[sort_idx[i]], ids_options_list[sort_idx[i]].numel()])
            else:
                scores.append([options[sort_idx[i]], probs[sort_idx[i]]])
        else:
            if ids_options_list:
                scores.append(
                    [options[sort_idx[i]], options_log_prob[sort_idx[i]], ids_options_list[sort_idx[i]].numel()])
            else:
                scores.append([options[sort_idx[i]], options_log_prob[sort_idx[i]]])

    if use_scale:
        if ids_options_list:
            print(
                tabulate(scores, headers=["options", f"{title} (1e{int(min_scale)})", "num token"], tablefmt="pretty"))
        else:
            print(tabulate(scores, headers=["options", f"{title} (1e{int(min_scale)})"], tablefmt="pretty"))
    else:
        if ids_options_list:
            print(tabulate(scores, headers=["options", title, "num token"], tablefmt="pretty"))
        else:
            print(tabulate(scores, headers=["options", title], tablefmt="pretty"))


@Evaluate.register("cbt")
class CBT(Evaluate):
    def __init__(self, task: str = 'CN', detokenize: bool = False, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.raw_dataset = load_dataset("cbt", task, num_proc=4, split='test')
        self.verbose = verbose
        self.incorrect_idx = []
        self.detokenize = detokenize

    @torch.inference_mode()
    def compute(self, model, tokenizer, stop=None):
        model = model.eval()
        accuracy = 0
        model.eval()
        for idx, sample in tqdm(enumerate(self.raw_dataset), total=len(self.raw_dataset)):
            if stop is not None and idx > stop:
                break

            if self.detokenize:
                sentences = preprocess(sample['sentences'])
                question = preprocess(sample['sentences'])
            else:
                sentences = sample['sentences']
                question = sample['sentences']
            options = sample['options']
            answer = sample['answer']

            prompt = ''.join(sentences) + ''.join(question)
            inputs_prompt = tokenizer(prompt, return_tensors="pt").to('cuda')
            ids_options_list = [tokenizer(opt, return_tensors="pt").to('cuda').input_ids for opt in options]

            if len(inputs_prompt.input_ids[0]) + max([i.shape[1] for i in ids_options_list]) > 1024:
                warnings.warn('contex larger than 1024 is not supported!')
                continue
            options_log_prob = torch.zeros(len(ids_options_list)).to('cuda')
            log_prob = model(**inputs_prompt).logits.log_softmax(-1)
            for i, ids_options in enumerate(ids_options_list):
                if len(ids_options[0]) > 1:
                    ids_prompt = inputs_prompt.input_ids
                    log_prob_ = model(ids_prompt).logits.log_softmax(-1)
                    for id_option in ids_options[0]:
                        options_log_prob[i] += log_prob_[0, -1, id_option]
                        ids_prompt = torch.cat((ids_prompt, id_option[None, None]), dim=1)
                        log_prob_ = model(ids_prompt).logits.log_softmax(-1)

                    options_log_prob[i] = options_log_prob[i] / len(ids_options[0])
                else:
                    options_log_prob[i] = log_prob[0, -1, ids_options[0]]

            if options[options_log_prob.argmax()] == answer:
                accuracy += 1
            elif self.verbose:
                print(f"{idx}) Answer: {answer}")
                show_result(options, options_log_prob, ids_options_list)
                self.incorrect_idx.append(idx)
            else:
                self.incorrect_idx.append(idx)

        return accuracy

    @torch.inference_mode()
    def compute_loss_based(self, model, tokenizer, stop=None):
        accuracy = 0
        model.eval()
        for idx, sample in tqdm(enumerate(self.raw_dataset), total=len(self.raw_dataset)):

            if stop is not None and idx > stop:
                break
            sentences = sample['sentences']
            question = sample['question']
            options = sample['options']
            answer = sample['answer']

            prompt = ''.join(sentences) + ' ' + tokenizer.bos_token + ' ' + ''.join(question)
            inputs_prompt = tokenizer(prompt, return_tensors="pt").to('cuda')
            if len(inputs_prompt.input_ids[0]) > 1020:
                warnings.warn('contex larger than 1024 is not supported!')
                continue

            loss_options = torch.zeros(len(options))
            for i in range(len(options)):
                inputs = tokenizer(prompt.replace('XXXXX', options[i]), return_tensors="pt").to('cuda')
                output = model(**inputs, labels=inputs.input_ids)
                loss_options[i] = output.loss.item()

            if options[loss_options.argmin()] == answer:
                accuracy += 1
            elif self.verbose:
                print(f"{idx}) Answer: {answer}")
                show_result(options, loss_options, ids_options_list=None)
            else:
                self.incorrect_idx.append(idx)

        return accuracy / idx

    @torch.inference_mode()
    def compute_alter(self, model, tokenizer, stop=None):
        accuracy = 0
        model.eval()
        for idx, sample in tqdm(enumerate(self.raw_dataset), total=len(self.raw_dataset)):

            if stop is not None and idx > stop:
                break
            if self.detokenize:
                sentences = preprocess(sample['sentences'])
                question = preprocess(sample['sentences'])
            else:
                sentences = sample['sentences']
                question = sample['sentences']
            options = sample['options']
            answer = sample['answer']

            prompt = ''.join(sentences) + ' ' + tokenizer.bos_token + ' ' + ''.join(question)
            prompt_tokens_lenght = tokenizer(prompt, return_tensors="pt")['input_ids'].shape[1]
            options_log_prob = []
            if prompt_tokens_lenght > 1020:
                warnings.warn('contex larger than 1024 is not supported!')
                continue

            ids_options_list = []
            for opt in options:
                inputs = tokenizer(prompt + opt, return_tensors="pt").to('cuda')
                log_prob = model(**inputs).logits.log_softmax(-1)  # [..., :-1, :].contiguous()
                option_tokens_lenght = inputs['input_ids'].shape[1] - prompt_tokens_lenght
                prompt_token_ids = inputs['input_ids'][0, -option_tokens_lenght:]
                log_prob_ = 0
                for i in range(option_tokens_lenght):
                    log_prob_ += log_prob[0, -(i + 1), prompt_token_ids[i]]
                log_prob_ = log_prob_ / option_tokens_lenght
                options_log_prob.append(log_prob_.item())
                ids_options_list.append(prompt_token_ids)

            if options[torch.tensor(options_log_prob).argmax()] == answer:
                accuracy += 1
                print('accuracy', accuracy)
            elif self.verbose:
                print(f"{idx}) Answer: {answer}")
                show_result(options, torch.tensor(options_log_prob), ids_options_list)
                self.incorrect_idx.append(idx)
            else:
                self.incorrect_idx.append(idx)

        return accuracy / idx

    def compute_single(self, model, tokenizer, idx):
        model.eval()
        sample = self.raw_dataset[idx]
        sentences = sample['sentences']
        question = sample['question']
        options = sample['options']
        answer = sample['answer']

        prompt = ''.join(sentences) + ''.join(question)
        inputs_prompt = tokenizer(prompt, return_tensors="pt").to('cuda')
        ids_options_list = [tokenizer(opt, return_tensors="pt").to('cuda').input_ids for opt in options]
        ids_answer = tokenizer(answer, return_tensors="pt").to('cuda').input_ids

        if len(inputs_prompt.input_ids[0]) + max([i.shape[1] for i in ids_options_list]) > 1024:
            print('ctx>1024!!!!!!')
            return 0
        options_log_prob = torch.zeros(len(ids_options_list)).to('cuda')
        log_prob = model(**inputs_prompt).logits.log_softmax(-1)
        for i, ids_options in enumerate(ids_options_list):
            if len(ids_options[0]) > 1:
                ids_prompt = inputs_prompt.input_ids
                log_prob_ = model(ids_prompt).logits.log_softmax(-1)
                for id_option in ids_options[0]:
                    options_log_prob[i] += log_prob_[0, -1, id_option]
                    ids_prompt = torch.cat((ids_prompt, id_option[None, None]), dim=1)
                    log_prob_ = model(ids_prompt).logits.log_softmax(-1)

                options_log_prob[i] = options_log_prob[i] / len(ids_options[0])
            else:
                options_log_prob[i] = log_prob[0, -1, ids_options[0]]

        print(f"{idx}) Answer: {answer}")
        show_result(options, options_log_prob, ids_options_list)

    def compute_single_alter(self, model, tokenizer, idx):
        model.eval()
        sample = self.raw_dataset[idx]
        sentences = sample['sentences']
        question = sample['question']
        options = sample['options']
        answer = sample['answer']

        prompt = ''.join(sentences) + ' ' + tokenizer.bos_token + ' ' + ''.join(question)
        prompt_tokens_length = tokenizer(prompt, return_tensors="pt")['input_ids'].shape[1]
        options_log_prob = []
        if prompt_tokens_length > 1020:
            warnings.warn('contex larger than 1024 is not supported!')
            return 0

        ids_options_list = []
        for opt in options:
            inputs = tokenizer(prompt + opt, return_tensors="pt").to('cuda')
            log_prob = model(**inputs).logits.log_softmax(-1)  # [..., :-1, :].contiguous()
            option_tokens_lenght = inputs['input_ids'].shape[1] - prompt_tokens_length
            prompt_token_ids = inputs['input_ids'][0, -option_tokens_lenght:]
            log_prob_ = 0
            for i in range(option_tokens_lenght):
                log_prob_ += log_prob[0, -(i + 1), prompt_token_ids[i]]
            log_prob_ = log_prob_ / option_tokens_lenght
            options_log_prob.append(log_prob_.item())
            ids_options_list.append(prompt_token_ids)

        print(f"{idx}) Answer: {answer}")
        show_result(options, torch.tensor(options_log_prob), ids_options_list)

    def compute_single_loss_based(self, model, tokenizer, idx):
        model.eval()
        sample = self.raw_dataset[idx]
        sentences = sample['sentences']
        question = sample['question']
        options = sample['options']
        answer = sample['answer']

        prompt = ''.join(sentences) + ' ' + tokenizer.bos_token + ' ' + ''.join(question)
        inputs_prompt = tokenizer(prompt, return_tensors="pt").to('cuda')
        if len(inputs_prompt.input_ids[0]) > 1020:
            warnings.warn('contex larger than 1024 is not supported!')

        loss_options = torch.zeros(len(options))
        for i in range(len(options)):
            inputs = tokenizer(prompt.replace('XXXXX', options[i]), return_tensors="pt").to('cuda')
            output = model(**inputs, labels=inputs.input_ids)
            loss_options[i] = output.loss.item()

        print(f"{idx}) Answer: {answer}")
        show_result(options, loss_options, ids_options_list=None, use_scale=False, title='loss')

    def show_question(self, idx):

        from rich.console import Console
        from sacremoses import MosesTokenizer, MosesDetokenizer
        detokenizer = MosesDetokenizer(lang='en')

        sample = self.raw_dataset[idx]
        sentences = sample['sentences']
        question = sample['question']
        options = sample['options']
        answer = sample['answer']
        console = Console()

        console.print(f"sentences: \n[italic]{''.join(detokenizer.detokenize(sentences))}[/italic]")
        console.print(f"question: \n[italic]{''.join(detokenizer.detokenize(question.split()))}[/italic]")
        options = [f"[bold]{_}[/bold]" if _ == answer else _ for _ in options]
        console.print(f"options: \n[italic]{''.join(detokenizer.detokenize(options))}[/italic]")
