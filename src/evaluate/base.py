from common import FromParams, Registrable, Params
import torch
from tqdm import tqdm

class Evaluate(Registrable):
    def __init__(self, num_beams: int = 5, top_k: int = 100, top_p: float = 0.9, do_sample: bool = False):
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams

# code from: https://huggingface.co/docs/transformers/perplexity
def compute_perplexity(model, tokenizer, raw_dataset, stride):

    encodings = tokenizer("\n\n".join(raw_dataset["text"]), return_tensors="pt")
    max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
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
