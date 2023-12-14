from common import FromParams, Registrable, Params


class Evaluate(Registrable):
    def __init__(self, num_beams: int = 5, top_k: int = 100, top_p: float = 0.9, do_sample: bool = False):
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams
