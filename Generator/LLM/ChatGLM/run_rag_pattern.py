from vllm import LLM,SamplingParams
import os
import json

# TEST_DATA = 'hearthstone_bm25'
TEST_DATA = 'concode_bm25'
# TEST_DATA = 'conala_bm25'

if 'hearthstone' in TEST_DATA:
    output_tokens_length = 700
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif 'concode' in TEST_DATA:
    output_tokens_length = 300
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
elif 'conala' in TEST_DATA:
    output_tokens_length = 100
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
print('Outputh Length is: ',output_tokens_length)

TEST_DATA_PATH = TEST_DATA.split('_')[0]
     
file_name = "../dataset/{}/nl_test.json".format(TEST_DATA)

llm = LLM(model="THUDM/chatglm3-6b",trust_remote_code=True,dtype='half')  # Create an LLM.
sampling_params = SamplingParams(temperature=0.85, top_p=0.95,max_tokens=output_tokens_length)

if not os.path.exists("./RAG/pattern/{}/".format(TEST_DATA_PATH)):
    os.makedirs("./RAG/pattern/{}/".format(TEST_DATA_PATH))

with open(file_name, 'r') as file,open("./RAG/pattern/{}/{}_output_RAG_pattern.txt".format(TEST_DATA_PATH,TEST_DATA), 'a') as txt_file,open("./RAG/pattern/{}/{}_gold_RAG_pattern.txt".format(TEST_DATA_PATH,TEST_DATA), 'a') as gold:
    for i,data in enumerate(file.readlines()):
        data = json.loads(data)
        source = data["nl"].split('retrieved_nl')
        # print(source)
        # print(len(source))
        # print(source[1].split('retrieved_code'))
        tgt = data["code"]

        if 'hearthstone' in TEST_DATA:
            prompt = "<NL 1> \n {} </NL 1> \n \
                  <Code Snippet 1> \n {} </Code Snippet 1> \n \
                  <NL 2> \n {} </NL 2> \n \
                  <Code Snippet 2> \n {} </Code Snippet 2> \n \
                  <NL> \n {} </NL> \n \
                  <Code Snippet> \n".format(source[1].split('retrieved_code')[0].strip(),source[1].split('retrieved_code')[1].strip(),source[2].split('retrieved_code')[0].strip(),source[2].split('retrieved_code')[1].strip(),source[0].strip())
        else:
            nl1 = "<NL 1> \n {} </NL 1> \n <Code Snippet 1> \n {} </Code Snippet 1> \n".format(source[1].split('retrieved_code')[0].strip(),source[1].split('retrieved_code')[1].strip())
            nl2 = "<NL 2> \n {} </NL 2> \n <Code Snippet 2> \n {} </Code Snippet 2> \n".format(source[2].split('retrieved_code')[0].strip(),source[2].split('retrieved_code')[1].strip())
            nl3 = "<NL 3> \n {} </NL 3> \n <Code Snippet 3> \n {} </Code Snippet 3> \n".format(source[3].split('retrieved_code')[0].strip(),source[3].split('retrieved_code')[1].strip())
            nl4 = "<NL 4> \n {} </NL 4> \n <Code Snippet 4> \n {} </Code Snippet 4> \n".format(source[4].split('retrieved_code')[0].strip(),source[4].split('retrieved_code')[1].strip())
            nl5 = "<NL 5> \n {} </NL 5> \n <Code Snippet 5> \n {} </Code Snippet 5> \n".format(source[5].split('retrieved_code')[0].strip(),source[5].split('retrieved_code')[1].strip())
            nl  = "<NL> \n {} </NL> \n <Code Snippet>".format(source[0].strip())
            prompt = nl1 + nl2 + nl3 + nl4 + nl5 + nl
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