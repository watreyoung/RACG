from vllm import LLM,SamplingParams
import os
import json

TEST_DATA = 'hearthstone_bm25'
TEST_DATA = 'concode_bm25'
# TEST_DATA = 'conala_bm25'

if 'hearthstone' in TEST_DATA:
    output_tokens_length = 700
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
elif 'concode' in TEST_DATA:
    output_tokens_length = 300
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
elif 'conala' in TEST_DATA:
    output_tokens_length = 100
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
print('Outputh Length is: ',output_tokens_length)

file_name = "../dataset/{}/test.json".format(TEST_DATA)

llm = LLM(model="codellama/CodeLlama-7b-Instruct-hf",trust_remote_code=True,dtype='half')  # Create an LLM.
sampling_params = SamplingParams(temperature=0.9, top_p=0.95,max_tokens=output_tokens_length)

with open(file_name, 'r') as file,open("{}_output_baseline.txt".format(TEST_DATA), 'a') as txt_file,open("{}_gold_baseline.txt".format(TEST_DATA), 'a') as gold:
    for i,data in enumerate(file.readlines()):
        data = json.loads(data)
        source = data["nl"].split('retrieved_code')[0]
        tgt = data["code"]

        prompt = "You are a expert in software engineering and development, and you process a strong proficiency in translating natural language descriptions into code snippets that meet the specified requirements. " + \
                 "Here is the natural language description: # {} #. Please provide the corresponding code snippet that fulfills the given description directly and do not output other information.".format(source)
                 
        outputs = llm.generate(prompt,sampling_params)  # Generate texts from the prompts.
        # print(outputs)
        if outputs == []:
            txt_file.write('The question is from {}\n'.format(str(i)))
        else:
            output = outputs[0]
            re = output.outputs[0].text
            re = re.replace('\n',' ').replace('\r',' ')
            txt_file.write(f'{re}\n')
        gold.write(str(tgt).replace('\n',' ').replace('\r',' '))
        gold.write('\n')