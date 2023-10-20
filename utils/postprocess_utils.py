import os
import re
import tiktoken

from .constants import *

def direct_postprocess(engine, table_id_list, response_list, keywords):
    write_data = {}
    for (id, res) in zip(table_id_list, response_list):
        claim_list = []
        if engine == CHAT_GPT or engine == GPT4:
            claims = res.choices[0].message['content']
        else:
            claims = res.choices[0]['text']

        include_keyword = False    
        for claim in claims.split('\n'):
            for keyword in keywords:
                if keyword in claim:
                    include_keyword = True
                    claim = claim.replace(keyword, '').strip().replace('</s>', '')
                    claim_list.append(claim)
        if include_keyword == False:
            claims = claims.replace('\n', '').strip().replace('</s>', '')
            claim_list.append(claims)
        if len(claim_list) == 1:
            write_data[id] = claim_list[0]
        else:
            write_data[id] = claim_list
    return write_data


def improve_postprocess(engine, responses, keywords, ori_sent):
    pred_vote = {'Entailed': [], 'Refuted': []}
    for res in responses:
        if engine == CHAT_GPT or engine == GPT4:
            feedback = res.choices[0].message['content']
        elif engine == DAVINCI003 or engine == DAVINCI002:
            feedback = res.choices[0]['text']
        else:
            feedback = res

        if 'no error' in feedback or 'No error' in feedback or 'No Error' in feedback:
            pred_vote['Entailed'].append(ori_sent)
        else:
            for sentence in feedback.split('\n'):
                if_error = False
                if any([keyword in sentence for keyword in keywords]):
                    if_error = True
                    new_sentence_with_prefix = sentence
                    break
            if if_error:
                new_sentence = re.sub(r'^[\W_]+|[\W_]+$','',str(new_sentence_with_prefix.split(":")[1]).strip())
                if len(new_sentence) == 0:
                    new_sentence = feedback.split('New answer:')[-1].strip()
                pred_vote['Refuted'].append(new_sentence)
            else:
                conclusion = feedback.split('\n')[-1]
                pred_vote['Refuted'].append(conclusion)
    return pred_vote
    

def get_exact_output_path(output_path, engine, dataset, mode, direct_mode=None, finetuned_model=None):
    open_src_model = ['llama-7b', 'llama-13b', 'llama-30b', 'llama-65b', 'llama2-70b', 'vicuna', 'tulu', 'alpaca', 'pythia']
    if engine == CHAT_GPT:
        engine_name = 'GPT3.5'
    elif engine == GPT4:
        engine_name = 'GPT4'
    elif engine == DAVINCI003:
        engine_name = 'Davinci003'
    elif engine == DAVINCI002:
        engine_name = 'Davinci002'
    elif engine in open_src_model:
        engine_name = engine
    else:
        raise ValueError("Invalid engine name")
    
    if mode == 'direct':
        output_path = os.path.join(output_path, f"{dataset}_{engine_name}_direct_output_{direct_mode}.json")
    elif mode == 'improve':
        output_path = os.path.join(output_path, f"{finetuned_model}_{dataset}_{engine_name}_output_improved.json")
    else:
        raise ValueError("Invalid mode name")
    return output_path


def table_length_is_valid(input_table_str):
    enc = tiktoken.encoding_for_model("gpt-4")
    input_token_len = len(enc.encode(input_table_str))
    return input_token_len


def FeTaQA_F2WTQ_CoT_clean(data_dict):
    output_data = data_dict
    keywords = ["looking at", "Looking at", "Reasoning:", "reasoning:"]
    for csv_id, sent in data_dict.items():
        if table_length_is_valid(sent) > 30:
            for keyword in keywords:
                if keyword in sent:
                    if sent.count('.') > 1:
                        if keyword in sent.split('.')[-1]:
                            sent = sent.split('.')[0]
                        elif sent[-1] == '.':
                            sent = sent.split('.')[-2]
                        else:
                            sent = sent.split('.')[-1]
                    else:
                        if keyword in sent.split(',')[-1]:
                            sent = sent.split(',')[0]
                        else:
                            sent = sent.split(',')[-1]
                    output_data[csv_id] = sent
                    break
        else:
            output_data[csv_id] = sent.replace("Reasoning: ", '').replace("reasoning: ", '').replace("Looking at ", '').replace("looking at ", '')
    return output_data