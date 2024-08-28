from transformers import RobertaConfig,RobertaTokenizer,RobertaModel
import torch
import os
import numpy as np
import multiprocessing
import pickle
import json
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='3'

from myrun import InputFeatures
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

BASE_MODEL = "DeepSoftwareAnalytics/CoCoSoDa"
DATA = "conala"
MODEL_PATH = "./saved_models/fine_tune/{}/".format(DATA)
FILE_PATH = "./dataset/{}/".format(DATA)
TRAIN_FILE_NAME = os.path.join(FILE_PATH,"{}-train.json".format(DATA))
TEST_FILE_NAME = os.path.join(FILE_PATH,"{}-test.json".format(DATA))
DEV_FILE_NAME = os.path.join(FILE_PATH,"{}-dev.json".format(DATA))

EVAL_BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pool = multiprocessing.Pool(16)

config = RobertaConfig.from_pretrained(BASE_MODEL)
tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)
model = RobertaModel.from_pretrained(BASE_MODEL)

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


def convert_examples_to_features(js):
    js,tokenizer,args=js

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
    def __init__(self, tokenizer, file_path=None,pool=None):
        if 'train' in file_path:
            self.split = "train"
        else:
            self.split = "other"
        self.examples = []
        data=[]
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                data.append((js,tokenizer))

            self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                print("*** Example ***")
                print("idx: {}".format(idx))
                print("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                print("code_ids: {}".format(' '.join(map(str, example.code_ids))))             
                print("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))      
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].code_ids),
                    torch.tensor(self.examples[item].nl_ids))
     
 


def evaluate(model, tokenizer,file_name,pool, eval_when_training=False):
    dataset_class = TextDataset
    query_dataset = dataset_class(tokenizer, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=EVAL_BATCH_SIZE,num_workers=4)
    code_dataset = dataset_class(tokenizer, file_name, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=EVAL_BATCH_SIZE,num_workers=4)    
    # Eval!
    print("***** Running evaluation on java *****")
    print("  Num queries = %d", len(query_dataset))
    print("  Num codes = %d", len(code_dataset))
    print("  Batch size = %d", EVAL_BATCH_SIZE)

    model.eval()
    model_eval = model.module if hasattr(model,'module') else model
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        nl_inputs = batch[-1].to(device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            print(nl_vec)
            print(nl_inputs)

            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        with torch.no_grad():
            code_inputs = batch[0].to(device)
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())  

    model.train()    
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)
    scores=np.matmul(nl_vecs,code_vecs.T)
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    nl_tokens=[]
    code_tokens=[]
    for example in query_dataset.examples:
        nl_tokens.append(' '.join(example.nl_tokens))
        
    for example in code_dataset.examples:
        code_tokens.append(' '.join(example.code_tokens))
        
    ranks=[]
    for nl_token, sort_id in zip(nl_tokens,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_tokens[idx]==nl_token:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    # if args.save_evaluation_reuslt:
    #     evaluation_result = {"nl_urls":nl_urls, "code_urls":code_urls,"sort_ids":sort_ids[:,:10],"ranks":ranks}
    #     save_pickle_data(args.save_evaluation_reuslt_dir, "evaluation_result.pkl",evaluation_result)
    result = cal_r1_r5_r10(ranks)
    result["eval_mrr"]  = round(float(np.mean(ranks)),3)
    return result



for index in range(5):
        print("reload pytorch model from {}".format(MODEL_PATH))
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH,"{}/model.bin".format(index))),strict=False)
        model.to(device)
        evaluate(model,tokenizer,TEST_FILE_NAME,pool)