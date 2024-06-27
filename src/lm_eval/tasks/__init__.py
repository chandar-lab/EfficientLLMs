from . import superglue
from . import glue
from . import arc
from . import race
from . import webqs
from . import anli
from . import hellaswag
from . import lambada
from . import perplexity_tasks
from .common import HFTask

TASK_REGISTRY = {
    # GLUE
    "cola": glue.CoLA,
    "mnli": glue.MNLI,
    "mrpc": glue.MRPC,
    "rte": glue.RTE,
    "qnli": glue.QNLI,
    "qqp": glue.QQP,
    "stsb": glue.STSB,
    "sst": glue.SST,
    "wnli": glue.WNLI,
    # SuperGLUE
    "boolq": superglue.BoolQ,
    "commitmentbank": superglue.CommitmentBank,
    "copa": superglue.Copa,
    "multirc": superglue.MultiRC,
    "wic": superglue.WordsInContext,
    "wsc": superglue.WinogradSchemaChallenge,
    # Perplexity tasks
    "wikitext_2": perplexity_tasks.WikiText2,
    "wikitext_103": perplexity_tasks.WikiText103,
    "ptb": perplexity_tasks.PTB,
    "1bw": perplexity_tasks.OneBillionWord,
    # Order by benchmark/genre?
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    "race": race.RACE,
    "webqs": webqs.WebQs,
    "anli_r1": anli.ANLIRound1,
    "anli_r2": anli.ANLIRound2,
    "anli_r3": anli.ANLIRound3,
    "hellaswag": hellaswag.HellaSwag,
    "lambada": lambada.Lambada,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    return TASK_REGISTRY[task_name]


def get_task_dict(task_name_list):
    return {
        task_name: get_task(task_name)()
        for task_name in task_name_list
    }
