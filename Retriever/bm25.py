import numpy as np
import json
import os
from rank_bm25 import BM25Okapi
from transformers import RobertaTokenizer
import random

# 利用bm25算法得到top10个代码，得到的文件组织方式形如fid的输入文件
# 从训练集中检索
# 训练集进行检索时，相同id不作为检索结果
def bm25_top10(type = "test", ):
    retrieve_database = "./data/concode/train.json"
    source_file = "./data/concode/{}.json".format(type)
    target_file = "./data/concode/bm25_top10/{}.json".format(type)
    corpus_nl = [ ]
    corpus_code = [ ] 
    with open(retrieve_database, 'r')as f:
        for line in f:
            x = json.loads(line)
            corpus_nl.append(x['input'])
            corpus_code.append(x['output'])
    tokenized_corpus = [doc.split(" ") for doc in corpus_nl]
    bm25 = BM25Okapi(tokenized_corpus)
    results = [ ]
    datas = []
    with open(source_file,'r') as f1:
        for line in f1:
            x = json.loads(line)
            datas.append(x)
    print(str(len(datas)))
    for idx, x in enumerate(datas):
        query = x["input"].strip()
        target = x["output"].strip()
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)   # 计算query与corpus中每一条nl的得分
        n = 10 #前十条
        top = bm25.get_top_n(tokenized_query, corpus_nl, n = n+n)
        top_id = np.argsort(doc_scores)[::-1][:n+n]   # 寻找top n的id，此id从0起
        result = {}
        result['question'] = query
        result['target']= target
        result['answers']= [target]
        result['ctxs'] = []
        i = 0
        total = 0
        while total < 10:
            if type != "train" or top_id[i] != idx:  #train集检索时，跳过检索库中相同的id
                result['ctxs'].append({
                    'id':str(top_id[i]), 'title':top[i], 'text':corpus_code[top_id[i]], 'score':str(doc_scores[top_id[i]])  })
                total += 1
            i += 1
        results.append(result)      
        print(idx)     
    with open(target_file, "w") as f2:
        json.dump(results, f2, indent=2)


def top10_to_cat5(type = "test", cat_num = 5):
    source_file = "./data/concode/bm25_top10/{}.json".format(type)
    target_file = "./data/concode/bm25_cat{}/{}.json".format(str(cat_num), type)
    with open(source_file,'r') as f:
        retrieved_datas = json.load(f)
    f2 = open(target_file, 'w+', encoding="utf-8")
    tokenizer = RobertaTokenizer.from_pretrained('./pretrained_models/codet5_base')
    avg_src_len_tokenize = []
    for retrieved_data in retrieved_datas:
        x = {}
        x["code"] = retrieved_data['target']
        x["nl"] = retrieved_data['question'] 
        for i in range(cat_num):
            x["nl"] += " retrieved_code " + retrieved_data['ctxs'][i]['text'] 
        f2.write(json.dumps(x))
        f2.write('\n')
        avg_src_len_tokenize.append(len(tokenizer.tokenize(x["nl"])))
    print("[TOKENIZE] avg src len: {}, max src len: {}".format(np.mean(avg_src_len_tokenize), max(avg_src_len_tokenize)))


if __name__ == '__main__':
    bm25_top10("train")
    top10_to_cat5('train')