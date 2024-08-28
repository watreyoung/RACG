#!/usr/bin/env python
import os
import argparse


def get_cmd(task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch, warmup,
            model_dir, summary_dir, res_fn, max_steps=None, save_steps=None, log_steps=None):
    if max_steps is None:
        cmd_str = 'bash exp_with_args.sh %s %s %s %d %d %d %d %d %d %d %d %d %s %s %s' % \
                  (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
                   warmup, model_dir, summary_dir, res_fn)
    else:
        cmd_str = 'bash exp_with_args.sh %s %s %s %d %d %d %d %d %d %d %d %d %s %s %s %d %d %d' % \
                  (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
                   warmup, model_dir, summary_dir, res_fn, max_steps, save_steps, log_steps)
    return cmd_str


def get_args_by_task_model(task, sub_task, model_tag):
    if task == 'concode':
        # Read 100000 examples, avg src len: 71, avg trg len: 26, max src len: 567, max trg len: 140
        # [TOKENIZE] avg src len: 213, avg trg len: 33, max src len: 2246, max trg len: 264
        if sub_task == 'baseline':
            src_len = 320
            trg_len = 150
        elif sub_task in ['bm25', 'dpr', 'cocosoda', 'retromae']:
            src_len = 512
            trg_len = 150
        epoch = 3
        patience = 3
    elif  task == 'conala':
        # Read 2179 examples, avg src len: 10, avg trg len: 4, max src len: 29, max trg len: 29
        # [TOKENIZE] avg src len: 15, avg trg len: 16, max src len: 62, max trg len: 84
        if sub_task == 'baseline':
            src_len = 70
            trg_len = 100
        elif sub_task in ['bm25', 'dpr', 'cocosoda', 'retromae']:
            src_len = 300
            trg_len = 100
        epoch = 10
        patience = 10
    elif  task == 'hearthstone':
        # Read 533 examples, avg src len: 26, avg trg len: 21, max src len: 38, max trg len: 134
        # [TOKENIZE] avg src len: 72, avg trg len: 130, max src len: 114, max trg len: 635
        if sub_task == 'baseline':
            src_len = 150
            trg_len = 300
        elif sub_task in ['bm25', 'dpr', 'cocosoda', 'retromae']:
            src_len = 512
            trg_len = 300
        epoch = 30
        patience = 30

    if 'codet5_small' in model_tag:
        bs = 32
        if task == 'summarize' or task == 'translate' or (task == 'refine' and sub_task == 'small'):
            bs = 64
        elif task == 'clone':
            bs = 25
    elif 'codet5_large' in model_tag:
        bs = 10
    else:
        bs = 16

    if task == 'conala' :
        if sub_task == 'baseline':
            bs = 64
        elif sub_task in ['bm25', 'dpr', 'cocosoda', 'retromae']:
            bs = 32  
    if task == 'hearthstone':
        if sub_task == 'baseline':
            bs = 32
        elif sub_task in ['bm25', 'dpr', 'cocosoda', 'retromae']:
            bs = 16  
    lr = 10
    return bs, lr, src_len, trg_len, patience, epoch


def run_one_exp(args):
    bs, lr, src_len, trg_len, patience, epoch = get_args_by_task_model(args.task, args.sub_task, args.model_tag)
    print('============================Start Running==========================')
    cmd_str = get_cmd(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag, gpu=args.gpu,
                      data_num=args.data_num, bs=bs, lr=lr, source_length=src_len, target_length=trg_len,
                      patience=patience, epoch=epoch, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag))
    print('%s\n' % cmd_str)
    os.system(cmd_str) 


def get_sub_tasks(task):
    if task in ['concode', 'conala', 'hearthstone']:
        sub_tasks =  ['baseline', 'bm25', 'dpr', 'cocosoda', 'retromae']
    return sub_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='codet5_base',
                        choices=['roberta', 'codebert', 'bart_base', 'codet5_small', 'codet5_base', 'codet5_large'])
    parser.add_argument("--task", type=str, default='concode', choices=['concode', 'conala', 'hearthstone'])
    parser.add_argument("--sub_task", type=str, default='baseline', choices=['baseline', 'bm25', 'dpr', 'cocosoda', 'retromae' ])
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=int, default=3, help='index of the gpu to use in a cluster')
    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    run_one_exp(args)

