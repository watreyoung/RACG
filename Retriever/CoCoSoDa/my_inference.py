# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from unittest import removeResult
import torch.nn.functional as F
import argparse
import logging
import os
import pickle
import random
import torch
import json
from random import choice
import numpy as np
from itertools import cycle
from model import Model,Multi_Loss_CoCoSoDa
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)
from tqdm import tqdm
import multiprocessing
cpu_cont = 16
os.environ['CUDA_VISIBLE_DEVICES']='0'

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
import sys
sys.path.append("dataset")
from utils import save_json_data, save_pickle_data
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    

ruby_special_token = ['keyword', 'identifier', 'separators', 'simple_symbol', 'constant', 'instance_variable',
 'operator', 'string_content', 'integer', 'escape_sequence', 'comment', 'hash_key_symbol',
  'global_variable', 'heredoc_beginning', 'heredoc_content', 'heredoc_end', 'class_variable',]

java_special_token = ['keyword', 'identifier', 'type_identifier',  'separators', 'operator', 'decimal_integer_literal',
 'void_type', 'string_literal', 'decimal_floating_point_literal', 
 'boolean_type', 'null_literal', 'comment', 'hex_integer_literal', 'character_literal']

go_special_token = ['keyword', 'identifier', 'separators', 'type_identifier', 'int_literal', 'operator', 
'field_identifier', 'package_identifier', 'comment',  'escape_sequence', 'raw_string_literal',
'rune_literal', 'label_name', 'float_literal']

javascript_special_token =['keyword', 'separators', 'identifier', 'property_identifier', 'operator', 
'number', 'string_fragment', 'comment', 'regex_pattern', 'shorthand_property_identifier_pattern', 
'shorthand_property_identifier', 'regex_flags', 'escape_sequence', 'statement_identifier']

php_special_token =['text', 'php_tag', 'name', 'operator', 'keyword', 'string', 'integer', 'separators', 'comment', 
'escape_sequence', 'ERROR',  'boolean', 'namespace', 'class', 'extends']

python_special_token =['keyword', 'identifier', 'separators', 'operator', '"', 'integer', 
'comment', 'none', 'escape_sequence']


special_token={
    'python':python_special_token,
    'java':java_special_token,
    'ruby':ruby_special_token,
    'go':go_special_token,
    'php':php_special_token,
    'javascript':javascript_special_token
}

all_special_token = []
for key, value in special_token.items():
    all_special_token = list(set(all_special_token ).union(set(value)))

def lalign(x, y, alpha=2):
    x = torch.tensor(x)
    y= torch.tensor(y)
    return (x - y).norm(dim=1).pow(alpha).mean()
    # code2nl_pos = torch.einsum('nc,nc->n', [x, y]).unsqueeze(-1)

    # return code2nl_pos.mean()

def lunif(x, t=2):
    x = torch.tensor(x)
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

def cal_r1_r5_r10(ranks):
    r1,r5,r10= 0,0,0
    data_len= len(ranks)
    for item in ranks:
        if item >=1:
            r1 +=1
            r5 += 1 
            r10 += 1
        elif item >=0.2:
            r5+= 1
            r10+=1
        elif item >=0.1:
            r10 +=1
    result = {"R@1":round(r1/data_len,3), "R@5": round(r5/data_len,3),  "R@10": round(r10/data_len,3)}
    return result


#remove comments, tokenize code and extract dataflow                                        
def tokenizer_source_code(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
    except Exception as e:
        logger.error(e)
        dfg=[]
    return code_tokens

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                #  position_idx,
                #  dfg_to_code,
                #  dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        # self.position_idx=position_idx
        # self.dfg_to_code=dfg_to_code
        # self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids


def convert_examples_to_features(js):
    js,tokenizer,args=js

    parser=parsers["java"]
    parser=parsers["python"]

    # code
    code_tokens=tokenizer_source_code(js['code'],parser,args.lang)
    code_tokens=" ".join(code_tokens[:args.code_length-2])
    code_tokens=tokenizer.tokenize(code_tokens)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length   

    #nl
    nl=' '.join(js['nl'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length  

    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids)

  

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None):
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
        cache_file=args.output_dir+'/'+prefix+'.pkl'
        n_debug_samples = args.n_debug_samples
        # if 'codebase' in file_path:
        #     n_debug_samples = 100000
        if 'train' in file_path:
            self.split = "train"
        else:
            self.split = "other"
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
            if args.debug:
                self.examples= self.examples[:n_debug_samples]
        else:
            self.examples = []
            data=[]
            if args.debug:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data.append((js,tokenizer,args))
                        if len(data) >= n_debug_samples:
                            break
            else:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data.append((js,tokenizer,args))

            self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))             
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))      
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        if self.args.data_aug_type == "replace_type":
            return (torch.tensor(self.examples[item].code_ids),
                    torch.tensor(self.examples[item].code_type_ids),
                    torch.tensor(self.examples[item].nl_ids))
        else:
            return (torch.tensor(self.examples[item].code_ids),
                    torch.tensor(self.examples[item].nl_ids))
     
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all gpus
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, tokenizer,file_name,pool, eval_when_training=False):

    dataset_class = TextDataset
    query_dataset = dataset_class(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    # code_dataset = dataset_class(tokenizer, args, file_name, pool)
    code_dataset = dataset_class(tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation on %s *****"%args.lang)
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    model_eval = model.module if hasattr(model,'module') else model
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        nl_inputs = batch[-1].to(args.device)
        with torch.no_grad():
            if args.model_type ==  "base" :
                nl_vec = model(nl_inputs=nl_inputs) 

            elif args.model_type in  ["cocosoda" ,"no_aug_cocosoda", "multi-loss-cocosoda"]:
                outputs = model_eval.nl_encoder_q(nl_inputs, attention_mask=nl_inputs.ne(1))
                if args.agg_way == "avg":
                    outputs = outputs [0]
                    nl_vec = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                elif args.agg_way == "cls_pooler":
                    nl_vec =outputs [1]
                elif args.agg_way == "avg_cls_pooler":
                     nl_vec =outputs [1] +  (outputs[0]*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None] 
                nl_vec  = torch.nn.functional.normalize( nl_vec, p=2, dim=1)

            
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        with torch.no_grad():
            code_inputs = batch[0].to(args.device)
            if args.model_type ==  "base" :
                code_vec = model(code_inputs=code_inputs)
            elif args.model_type in  ["cocosoda" ,"no_aug_cocosoda", "multi-loss-cocosoda"]:
                # code_vec =  model_eval.code_encoder_q(code_inputs, attention_mask=code_inputs.ne(1))[1]
                outputs = model_eval.code_encoder_q(code_inputs, attention_mask=code_inputs.ne(1))
                if args.agg_way == "avg":
                    outputs = outputs [0]
                    code_vec  = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                elif args.agg_way == "cls_pooler":
                    code_vec=outputs [1]
                elif args.agg_way == "avg_cls_pooler":
                     code_vec=outputs [1] +  (outputs[0]*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None] 
                code_vec  = torch.nn.functional.normalize(code_vec, p=2, dim=1)
        
            code_vecs.append(code_vec.cpu().numpy())  

    model.train()    
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    scores=np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    res_nl=[]
    res_code=[]
    temp_code = {}
    temp_nl = []
    for example in query_dataset.examples:
        xxx = " ".join(example.nl_tokens)
        res_nl.append(xxx)
        temp_code[xxx] = " ".join(example.code_tokens)
    # print(res_nl[0])
        
    for example in code_dataset.examples:
        res_code.append(" ".join(example.code_tokens))
        temp_nl.append(" ".join(example.nl_tokens))
    # print(res_code[0])
    # print(res_nl[0]==temp_nl[0])
    # print(res_code[0]==temp_code[0])
        
    ranks=[]
    with open(os.path.join(args.output_dir,'4/cocosoda-test.json'),'w') as f:
        for nl, sort_id in zip(res_nl,sort_ids):
            line = {}
            line['nl'] = nl
            line['code'] = temp_code[nl]
            for idx in sort_id[:5]:
                if not temp_nl[idx] == nl:
                    line['nl'] += ' retrieved_code '
                    line['nl'] += res_code[idx]
            f.write(json.dumps(line))
            f.write('\n')

def parse_args():
    parser = argparse.ArgumentParser()
    # soda
    parser.add_argument('--data_aug_type',default="replace_type",choices=["replace_type", "random_mask" ,"other"], help="the ways of soda",required=False)
    parser.add_argument('--aug_type_way',default="random_replace_type",choices=["random_replace_type", "replace_special_type" ,"replace_special_type_with_mask"], help="the ways of soda",required=False)
    parser.add_argument('--print_align_unif_loss', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--do_ineer_loss', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--only_save_the_nl_code_vec', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--do_zero_short', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--agg_way',default="avg",choices=["avg", "cls_pooler","avg_cls_pooler" ], help="base is codebert/graphcoder/unixcoder",required=False)
    parser.add_argument('--weight_decay',default=0.01, type=float,required=False)
    parser.add_argument('--do_single_lang_continue_pre_train', action='store_true', help='do_single_lang_continue_pre_train', required=False)
    parser.add_argument('--save_evaluation_reuslt', action='store_true', help='save_evaluation_reuslt', required=False)
    parser.add_argument('--save_evaluation_reuslt_dir', type=str, help='save_evaluation_reuslt', required=False)
    parser.add_argument('--epoch', type=int, default=50,
                        help="random seed for initialization")
    # new continue pre-training
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--loaded_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument("--loaded_codebert_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument('--do_multi_lang_continue_pre_train', action='store_true', help='do_multi_lang_continue_pre_train', required=False)
    parser.add_argument("--couninue_pre_train_data_files", default=["dataset/ruby/train.jsonl",  "dataset/java/train.jsonl",], type=str, nargs='+', required=False,
                        help="The input training data files (some json files).")
    # parser.add_argument("--couninue_pre_train_data_files", default=["dataset/go/train.jsonl",  "dataset/java/train.jsonl",
    # "dataset/javascript/train.jsonl",  "dataset/php/train.jsonl",  "dataset/python/train.jsonl",  "dataset/ruby/train.jsonl",], type=list, required=False,
    #                     help="The input training data files (some json files).")
    parser.add_argument('--do_continue_pre_trained', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_fine_tune', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_whitening', action='store_true', help='do_whitening https://github.com/Jun-jie-Huang/WhiteningBERT', required=False)
    parser.add_argument("--time_score", default=1, type=int,help="cosine value * time_score")   
    parser.add_argument("--max_steps", default=100, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_warmup_steps", default=0, type=int, help="num_warmup_steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")    
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    # new moco
    parser.add_argument('--moco_type',default="encoder_queue",choices=["encoder_queue","encoder_momentum_encoder_queue" ], help="base is codebert/graphcoder/unixcoder",required=False)

    
    # debug
    parser.add_argument('--use_best_mrr_model', action='store_true', help='cosine_space', required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument("--max_codeblock_num", default=10, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--eval_frequency", default=1, type=int, required=False)
    parser.add_argument("--mlm_probability", default=0.1, type=float, required=False)

    # model type
    parser.add_argument('--do_avg', action='store_true', help='avrage hidden status', required=False)
    parser.add_argument('--model_type',default="base",choices=["base", "cocosoda","multi-loss-cocosoda","no_aug_cocosoda"], help="base is codebert/graphcoder/unixcoder",required=False)
    # moco
    # moco specific configs:
    parser.add_argument('--moco_dim', default=768, type=int,
                        help='feature dimension (default: 768)')
    parser.add_argument('--moco_k', default=32, type=int,
                        help='queue size; number of negative keys (default: 65536), which is divided by 32, etc.')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true',help='use mlp head')

    ## Required parameters
    parser.add_argument("--train_data_file", default="dataset/java/train.jsonl", type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default="saved_models/pre-train", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default="dataset/java/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/java/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="dataset/java/codebase.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default="java", type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=50, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=100, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=0, type=int,
                        help="Optional Data Flow input sequence length after tokenization.",required=False) 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=4, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=3407,
                        help="random seed for initialization")  
        
    #print arguments
    args = parser.parse_args()
    return  args                     

def create_model(args,model,tokenizer, config=None):
    # logger.info("args.data_aug_type %s"%args.data_aug_type)
    # replace token with type
    if args.data_aug_type in ["replace_type" , "other"] and not args.only_save_the_nl_code_vec:
        special_tokens_dict = {'additional_special_tokens': all_special_token}
        logger.info(" new token %s"%(str(special_tokens_dict)))
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
  
    if (args.loaded_model_filename) and ("pytorch_model.bin" in args.loaded_model_filename):
        logger.info("reload pytorch model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename),strict=False) 
        # model.from_pretrain
    if args.model_type ==  "base" :
        model= Model(model)
    elif args.model_type ==  "multi-loss-cocosoda":
        model= Multi_Loss_CoCoSoDa(model,args, args.mlp)
    if (args.loaded_model_filename) and ("pytorch_model.bin" not in args.loaded_model_filename) :
        logger.info("reload model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename)) 
        # model.load_state_dict(torch.load(args.loaded_model_filename,strict=False)) 
        # model.from_pretrained(args.loaded_model_filename)  
    if (args.loaded_codebert_model_filename) :
        logger.info("reload pytorch model from {}".format(args.loaded_codebert_model_filename))
        model.load_state_dict(torch.load(args.loaded_codebert_model_filename),strict=False)   
    logger.info(model.model_parameters())


    return model

def main():
    
    args = parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    pool = multiprocessing.Pool(cpu_cont)

    # Set seed
    set_seed(args.seed)

    #build model

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 
    model=create_model(args,model,tokenizer,config)

    logger.info("Training/evaluation parameters %s", args)
    args.start_step = 0

    model.to(args.device)

    if args.do_test:

        logger.info("runnning test")
        checkpoint_prefix = '4/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  

        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        evaluate(args, model, tokenizer,args.test_data_file, pool)


if __name__ == "__main__":
    main()

