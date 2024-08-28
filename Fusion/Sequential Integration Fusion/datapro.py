from transformers import RobertaTokenizer
import numpy as np
import json


def calc_stats():
    tokenizer = RobertaTokenizer.from_pretrained('./pretrained_models/codet5_base')
    # source_file = "./data/concode/train.json"
    # source_file = "./data/conala/conala-train.json"
    source_file = "./data/hearthstone/hearthstone_bm25_cat5/train.json"
    avg_src_len = []
    avg_src_len_tokenize = []
    avg_trg_len = []
    avg_trg_len_tokenize = []
    with open(source_file,'r') as f1:
        for idx, line in enumerate(f1):
            x = json.loads(line)
            avg_src_len.append(len(x["nl"].split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(x["nl"])))
            avg_trg_len.append(len(x["code"].split()))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(x["code"])))
    print("avg src len: {}, min src len: {}, max src len: {}".format(np.mean(avg_src_len), min(avg_src_len), max(avg_src_len)))
    print("[TOKENIZE] avg src len: {}, min src len: {}, max src len: {}".format(np.mean(avg_src_len_tokenize), min(avg_src_len_tokenize), max(avg_src_len_tokenize)))
    print("[TOKENIZE] avg trg len: {}, min trg len: {}, max trg len: {}".format(np.mean(avg_trg_len_tokenize), min(avg_trg_len_tokenize), max(avg_trg_len_tokenize)))
   


def top10_to_cat(dataset = "conala", type = "test", cat_num = "5"):
    source_file = "./data/{}/{}_bm25/{}_bm25_top10.json".format(dataset, dataset, type)
    target_file = "./data/{}/{}_bm25/{}_bm25_cat{}/{}.json".format(dataset, dataset, dataset, cat_num, type)
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
            x["nl"] += " retrieved_code " + retrieved_data['ctxs'][i]['text']  #直接拼接，不做其他处理，在检索的函数中完成去重的操作
        f2.write(json.dumps(x))
        f2.write('\n')
        avg_src_len_tokenize.append(len(tokenizer.tokenize(x["nl"])))
    print("[TOKENIZE] avg src len: {}, max src len: {}".format(np.mean(avg_src_len_tokenize), max(avg_src_len_tokenize)))


if __name__ == '__main__':
    top10_to_cat(dataset = "conala", type = "dev", cat_num = 6)
    top10_to_cat(dataset = "conala", type = "test", cat_num = 6)
    top10_to_cat(dataset = "conala", type = "train", cat_num = 6)
    # calc_stats()
