import json


def create_sample_expansion(dataset,split_mode='train'):
    filename = '{}/{}'.format(dataset,split_mode)
    with open(filename+'.json','r') as f1, open(filename+'_sef.json','w') as f2:
        for line in f1:
            data = json.loads(line)
            temp_data = data.split(' retrieved_code ')
            nl = temp_data[0]
            if split_mode == 'train':
                for retrieved_code in temp_data[1:]:
                    new_data = data
                    new_data['nl'] = nl + ' retrieved_code ' + retrieved_code
                    f2.write(json.dumps(data)+'\n')
            else:
                new_data = data
                new_data['nl'] = nl + ' retrieved_code ' + temp_data[1]
                f2.write(json.dumps(data)+'\n')
