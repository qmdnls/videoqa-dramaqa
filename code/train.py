import torch
import numpy as np
from ignite.engine.engine import Engine, State, Events

from ckpt import get_model_ckpt, save_ckpt
from loss import get_loss
from optimizer import get_optimizer
from logger import get_logger, log_results, log_results_cmd

from utils import prepare_batch
from metric import get_metrics
from evaluate import get_evaluator, evaluate_once, evaluate_by_logic_level
from metric.stat_metric import StatMetric
from torch import nn

def get_trainer(args, model, loss_fn, optimizer):
    def update_model(trainer, batch):
        args.pretraining = False
        model.train()
        optimizer.zero_grad()
        net_inputs, target = prepare_batch(args, batch, model.module.vocab)
        #net_inputs, target = prepare_batch(args, batch, model.vocab)
        y_pred, char_pred, mask_pred = model(**net_inputs)
        batch_size = y_pred.shape[0] 

        '''
        # get person ground truth and compute character loss
        n_char = 21
        visual_char = net_inputs['filtered_visual'].view(batch_size, -1, 3)[:,:,0]
        char_target = visual_char.unsqueeze(2).view(batch_size, -1)
        char_target = char_target.view(-1)
        char_pred = char_pred.view(-1, n_char)
        character_loss = nn.CrossEntropyLoss(ignore_index=-1).cuda()(char_pred, char_target)
      
        # get ground truth labels and compute MLM loss
        vocab_size = mask_pred.size(-1)
        mask_target = net_inputs['labels']
        mask_target = mask_target.view(-1)
        mask_pred = mask_pred.view(-1, vocab_size)
        mlm_loss = nn.CrossEntropyLoss(ignore_index=-1).cuda()(mask_pred, mask_target)
        '''

        # compute QA loss
        loss, stats = loss_fn(y_pred, target)
        
        # compute total loss
        #loss = loss + 0.1 * character_loss + 0.1 * mlm_loss
        loss.backward()
        optimizer.step()
        return loss.item(), stats, batch_size, y_pred.detach(), target.detach()

    trainer = Engine(update_model)

    metrics = {
        'loss': StatMetric(output_transform=lambda x: (x[0], x[2])),
        'top1_acc': StatMetric(output_transform=lambda x: ((x[3].argmax(dim=-1) == x[4]).float().mean().item(), x[2]))
    }
    if hasattr(loss_fn, 'get_metric'):
        metrics = {**metrics, **loss_fn.get_metric()}

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer

def get_pretrainer(args, model, loss_fn, optimizer):
    def update_model(trainer, batch):
        args.pretraining = True
        model.train()
        optimizer.zero_grad()
        net_inputs, target = prepare_batch(args, batch, model.module.vocab)
        #net_inputs, target = prepare_batch(args, batch, model.vocab)
        y_pred, char_pred, mask_pred = model(**net_inputs)
        batch_size = y_pred.shape[0] 
        
        # get person ground truth and compute character loss
        n_char = 21
        visual_char = net_inputs['filtered_visual'].view(batch_size, -1, 3)[:,:,0]
        char_target = visual_char.unsqueeze(2).view(batch_size, -1)
        char_target = char_target.view(-1)
        char_pred = char_pred.view(-1, n_char)
        character_loss = nn.CrossEntropyLoss(ignore_index=-1).cuda()(char_pred, char_target)
      
        # get ground truth labels and compute MLM loss
        vocab_size = mask_pred.size(-1)
        mask_target = net_inputs['labels']
        mask_target = mask_target.view(-1)
        mask_pred = mask_pred.view(-1, vocab_size)
        mlm_loss = nn.CrossEntropyLoss(ignore_index=-1).cuda()(mask_pred, mask_target)
        
        # compute QA loss
        loss, stats = loss_fn(y_pred, target)
        
        # compute total loss
        loss = character_loss + mlm_loss
        
        loss.backward()
        optimizer.step()
        return loss.item(), stats, batch_size, y_pred.detach(), target.detach()

    trainer = Engine(update_model)

    metrics = {
        'loss': StatMetric(output_transform=lambda x: (x[0], x[2])),
        'top1_acc': StatMetric(output_transform=lambda x: ((x[3].argmax(dim=-1) == x[4]).float().mean().item(), x[2]))
    }
    if hasattr(loss_fn, 'get_metric'):
        metrics = {**metrics, **loss_fn.get_metric()}

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def train(args):
    args, model, iters, vocab, ckpt_available = get_model_ckpt(args)

    if ckpt_available:
        print("loaded checkpoint {}".format(args.ckpt_name))
    loss_fn = get_loss(args, vocab)
    optimizer = get_optimizer(args, model)

    pretrainer = get_pretrainer(args, model, loss_fn, optimizer)
    trainer = get_trainer(args, model, loss_fn, optimizer)

    metrics = get_metrics(args, vocab)
    evaluator = get_evaluator(args, model, loss_fn, metrics)

    logger = get_logger(args)

    @pretrainer.on(Events.STARTED)
    def on_training_started(engine):
        print("Begin Pretraining")

    @pretrainer.on(Events.ITERATION_COMPLETED)
    def log_iter_results(engine):
        log_results(logger, 'pretrain/iter', engine.state, engine.state.iteration)

    @pretrainer.on(Events.EPOCH_COMPLETED)
    def evaluate_epoch(engine):
        log_results(logger, 'pretrain/epoch', engine.state, engine.state.epoch)

    """
    @pretrainer.on(Events.COMPLETED)
    def unfreeze_language_model(engine):
        for param in model.module.language_model.base_model.parameters():
            param.requires_grad = True
    """

    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        print("Begin Training")

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_results(engine):
        log_results(logger, 'train/iter', engine.state, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_epoch(engine):
        log_results(logger, 'train/epoch', engine.state, engine.state.epoch)
        state = evaluate_once(evaluator, iterator=iters['val'])
        log_results(logger, 'valid/epoch', state, engine.state.epoch)
        log_results_cmd('valid/epoch', state, engine.state.epoch)
        save_ckpt(args, engine.state.epoch, engine.state.metrics['loss'], model, vocab)
        evaluate_by_logic_level(args, model, iterator=iters['val'])

    if args.pretrain_epochs > 0:
        pretrainer.run(iters['pretrain'], max_epochs=args.pretrain_epochs) 
    trainer.run(iters['train'], max_epochs=args.max_epochs)
