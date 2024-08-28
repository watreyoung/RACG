from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from _utils import *

logger = logging.getLogger(__name__)


def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    examples = read_examples(filename, args.data_num, args.task)

    if is_sample:
        examples = random.sample(examples, min(10000, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data


def get_filenames(data_root, task, sub_task, split=''):
    if task == 'concode':
        if sub_task == 'baseline':
            data_dir = '{}/concode'.format(data_root)
            train_fn = '{}/train.json'.format(data_dir)
            dev_fn = '{}/dev.json'.format(data_dir)
            test_fn = '{}/test.json'.format(data_dir)
        elif sub_task in ['bm25', 'dpr', 'cocosoda', 'retromae']:
            data_dir = '{}/{}_{}'.format(data_root, task, sub_task)
            train_fn = '{}/train.json'.format(data_dir)
            dev_fn = '{}/dev.json'.format(data_dir)
            test_fn = '{}/test.json'.format(data_dir)
    elif task == 'conala':
        if sub_task == 'baseline':
            data_dir = '{}/{}'.format(data_root, task)
            train_fn = '{}/conala-train.json'.format(data_dir)
            dev_fn = '{}/conala-dev.json'.format(data_dir)
            test_fn = '{}/conala-test.json'.format(data_dir)
        elif sub_task in ['bm25', 'dpr', 'cocosoda', 'retromae']:
            data_dir = '{}/{}/conala_{}/conala'.format(data_root, task, sub_task)
            train_fn = '{}/train.json'.format(data_dir)
            dev_fn = '{}/dev.json'.format(data_dir)
            test_fn = '{}/test.json'.format(data_dir)
    elif task == 'hearthstone':
        if sub_task == 'baseline':
            data_dir = '{}/{}'.format(data_root, task)
            train_fn = '{}/hearthstone-train.json'.format(data_dir)
            dev_fn = '{}/hearthstone-dev.json'.format(data_dir)
            test_fn = '{}/hearthstone-test.json'.format(data_dir)
        elif sub_task in ['bm25', 'dpr', 'cocosoda', 'retromae']:
            data_dir = '{}/{}/hearthstone_{}'.format(data_root, task, sub_task)
            train_fn = '{}/train.json'.format(data_dir)
            dev_fn = '{}/dev.json'.format(data_dir)
            test_fn = '{}/test.json'.format(data_dir)
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'concode': read_concode_examples,
        'conala': read_conala_examples,
        'hearthstone': read_hearthstone_examples
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
