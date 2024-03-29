import json

import torch
from ignite.engine.engine import Engine, State, Events
from metric.stat_metric import StatMetric

from ckpt import get_model_ckpt
from model import get_model
from loss import get_loss
from logger import log_results_cmd

from utils import prepare_batch
from metric import get_metrics


def get_evaluator(args, model, loss_fn):
    def _inference(evaluator, batch):
        model.eval()
        with torch.no_grad():
            qids = batch["qid"]
            net_inputs, _ = prepare_batch(args, batch, model.module.vocab)
            y_pred, char_pred, mask_pred = model(**net_inputs)
            print("Before argmax:", y_pred.size())
            y_pred = y_pred.argmax(dim=-1)  # + 1  # 0~4 -> 1~5
            print("After argmax:", y_pred.size())
            for qid, ans in zip(qids, y_pred):
                engine.answers[qid] = ans.item()

            return

    engine = Engine(_inference)
    engine.answers = {}

    return engine


def evaluate_once(evaluator, iterator):
    evaluator.run(iterator)
    return evaluator.answers


def infer(args):
    #split = args.split 
    split = 'test'

    args, model, iters, vocab, ckpt_available = get_model_ckpt(args)
    print("loaded:", args.ckpt_name)
    print("ckpt_available:", ckpt_available)
    if ckpt_available:
        print("loaded checkpoint {}".format(args.ckpt_name))
    loss_fn = get_loss(args, vocab)

    evaluator = get_evaluator(args, model, loss_fn)

    answers = evaluate_once(evaluator, iterator=iters[split])
    keys = sorted(list(answers.keys()))
    answers = [{"correct_idx": answers[key], "qid": key} for key in keys]
    path = str(args.data_path.parent / 'answers.json')
    with open(path, 'w') as f:
        json.dump(answers, f, indent=4)

    print("saved outcome at {}".format(path))
