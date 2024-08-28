import json
import random

def process_json(json_file, train_file, output_file):
    data = []
    train_data = []
    with open(json_file, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))

    with open(train_file, 'r') as f:
        for line in f.readlines():
            train_data.append(json.loads(line))

    new_data = []
    for index,item in enumerate(data):
        nl = item['nl']
        retrieved_codes = nl.split('retrieved_code')[1:]
        # print(retrieved_codes)

        code_snippets = []
        for code in retrieved_codes:
            for snippet in train_data:
                if snippet['code'].strip() == code.strip():
                    code_snippets.append(' retrieved_nl '+snippet['nl'].split('retrieved_code')[0]+' retrieved_code '+code.strip())

        if len(code_snippets) == 0:
            temp_data = train_data[random.randint(0,len(train_data)-1)]
            code_snippets.append(' retrieved_nl ' + temp_data['intent'].split('retrieved_code')[0] + ' retrieved_code ' + temp_data['snippet'].strip())
            print(index)

        # print(code_snippets)
        while len(code_snippets) < 5:
            code_snippets.append(code_snippets[0])
        code_snippets_str = ' '.join(code_snippets)
        item['nl'] = nl.split('retrieved_code')[0] + code_snippets_str
        new_data.append(item)

    with open(output_file, 'w') as f:
        for d in new_data:
            f.write(json.dumps(d)+'\n')

for TEST_DATA in ['concode_unixcoder','hearthstone_unixcoder']:
# for TEST_DATA in ['codebert_concode','concode_retromae']:
# for TEST_DATA in ['concode_reversed', 'hearthstone_reversed']:
# for TEST_DATA in ['concode_bm25', 'hearthstone_bm25']:
    json_file = '{}/test.json'.format(TEST_DATA)
    # json_file = '{}/random_augmented.json'.format(TEST_DATA)
    train_file = '{}/train.json'.format(TEST_DATA)
    output_file = '{}/nl_test.json'.format(TEST_DATA)
    process_json(json_file, train_file, output_file)

def process_conala_json(json_file, train_file, output_file):
    data = []
    train_data = []
    with open(json_file, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))

    with open(train_file, 'r') as f:
        for line in f.readlines():
            train_data.append(json.loads(line))

    new_data = []
    for index,item in enumerate(data):
        nl = item['nl']
        retrieved_codes = nl.split('retrieved_code')[1:]
        # print(retrieved_codes)

        code_snippets = []
        for code in retrieved_codes:
            for snippet in train_data:
                if snippet['snippet'].strip() == code.strip():
                    code_snippets.append(' retrieved_nl '+snippet['intent'].split('retrieved_code')[0]+' retrieved_code '+code.strip())

        if len(code_snippets) == 0:
            temp_data = train_data[random.randint(0,len(train_data)-1)]
            code_snippets.append(' retrieved_nl ' + temp_data['intent'].split('retrieved_code')[0] + ' retrieved_code ' + temp_data['snippet'].strip())
            print(index)

        while len(code_snippets) < 5:
            code_snippets.append(code_snippets[0])
        # print(code_snippets)
        code_snippets_str = ' '.join(code_snippets)
        item['nl'] = nl.split('retrieved_code')[0] + code_snippets_str
        new_data.append(item)

    with open(output_file, 'w') as f:
        for d in new_data:
            f.write(json.dumps(d)+'\n')

# json_file = 'conala_unixcoder/test.json'
# train_file = '/data/zzyang/RACG/dataset/conala-corpus-origin/conala-mined-nonrepeatintent.jsonl'
# output_file = 'conala_bm25/nl_test.json'
# process_conala_json(json_file, train_file, output_file)
