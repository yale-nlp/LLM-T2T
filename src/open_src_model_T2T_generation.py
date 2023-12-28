import sys
import argparse
import json
import os
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

sys.path.append('')
from utils import *

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

PREPROCESSING_FUNCTIONS = {
    "tulu": process_prompt_for_tulu,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def prompt_encoding(tokenizer, prompt_text, args):
    requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
        preprocessed_prompt_text = prepare_input(prompt_text)
        encoded_prompt = tokenizer(preprocessed_prompt_text, return_tensors="pt").input_ids
    else:
        encoded_prompt = tokenizer(prompt_text, return_tensors="pt").input_ids

    encoded_prompt = encoded_prompt.to("cuda")
    return encoded_prompt


def get_output_sequence(model, encoded_prompt, tokenizer, args, prompt_text):
    output_sequence = model.generate(
                    input_ids=encoded_prompt,
                    max_length=args.length + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token is None else tokenizer.pad_token_id,
                )[0]

    # Decode text
    text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)
    # Remove all text after the stop token
    text = text[text.find(prompt_text) + len(prompt_text):].split(args.stop_token)[0].strip()

    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
    total_sequence = text
    return total_sequence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help='Model type selected in the list: ["llama-7b", ""llama-13b", "llama-30b", "llama-65b", "llama2-70b, "vicuna-13b", "tulu-13b", "alpaca-13b", "pythia-12b"]')
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help='huggingface model name or path, can select from ["huggyllama/llama-7b", "huggyllama/llama-13b", "huggyllama/llama-30b", "huggyllama/llama-65b", "TheBloke/Llama-2-70B-fp16", "lmsys/vicuna-13b-v1.3", "TheBloke/tulu-13B-fp16", "TheBloke/gpt4-alpaca-lora-13B-HF", "EleutherAI/pythia-12b"]')
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help='which mode to use, can select from ["direct", "improve"]')
    parser.add_argument("--direct_mode", type=str, default='CoT',
                        help='which direct submode to use, only valid when you choose "direct" as mode, can select from ["CoT", "without_CoT"]')
    parser.add_argument("--prompt_path", type=str, default=None,
                        help='which prompt to use, only valid when you choose "direct" as mode. If not specified, we will use the default prompt for each direct mode')
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help='which dataset to use, can select from ["LogicNLG", "FeTaQA", "LoTNLG", "F2WTQ"]')
    parser.add_argument("--data_path", default=None, type=str,
                        help='original data path for the dataset. If not specified, we will use the default path for each dataset')
    parser.add_argument("--finetuned_model_path", type=str, default=None,
                        help='output path of finetuned small model for the dataset, only valid when you choose "improve" as mode')
    parser.add_argument("--length", type=int, default=256,
                        help="adjust length to model")
    parser.add_argument("--stop_token", type=str, default="\n#",
                        help="Token at which text generation is stopped")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature for LLM")
    parser.add_argument("--k", type=int, default=0,
                        help="k for top-k sampling")
    parser.add_argument("--p", type=float, default=0.9,
                        help="p for top-p sampling")
    parser.add_argument("--output_path", type=str, default="output/test_path",
                        help='output path for the generated data')
    parser.add_argument("--num_limit", type=int, default=10,
                        help='number of tables used to generate')
    parser.add_argument("--num_paths", type=int, default=3,
                        help='number of paths used to improve using self-consistency mechanism, only valid when you choose "improve" as mode')
    args = parser.parse_args()

    finetuned_model = None
    if "tulu" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory_mapping = {0: "46GB", 1: "46GB"}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map="auto",
        load_in_8bit=True,
        max_memory=max_memory_mapping if ("65" in args.model_name_or_path or "70" in args.model_name_or_path) else None,
    )

    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.length = adjust_length_to_model(args.length, max_sequence_length=max_seq_length)

    prompt_path = args.prompt_path    
    if prompt_path == None:
        if args.mode == 'direct':
            prompt_path = f'prompts/open_src_model/{args.dataset}/prompt_{args.dataset}_{args.mode}_{args.direct_mode}.txt'
        else:
            prompt_path = f'prompts/open_src_model/{args.dataset}/prompt_{args.dataset}_{args.mode}.txt'

    prompt_template = open(prompt_path, 'r', encoding='utf-8').read().strip()

    data_path = args.data_path
    if data_path == None:
        if args.dataset == 'LogicNLG':
            data_path = LOGICNLG_PATH
        elif args.dataset == 'FeTaQA':
            data_path = FETAQA_PATH
        elif args.dataset == 'LoTNLG':
            data_path = LOTNLG_PATH
        elif args.dataset == 'F2WTQ':
            data_path = F2WTQ_PATH
        else:
            raise ValueError("Invalid dataset!")
        
    data = read_json(data_path=data_path)
    output_data = {}
    num = 0

    if args.mode == 'improve' and args.dataset == 'LogicNLG':
        for cur_id, current in tqdm(data.items()):
            if num >= args.num_limit:
                break
            table_id = current['csv_id']
            title = current['title']
            table = current['table_text'].replace('<br>', '\n')
            sents = current['sentences']
        

            feedback_list = []

            for sent in sents:
                prompt_text = prompt_template.replace('{title}', title).replace('{table}', table).replace('{sent}', sent)
                encoded_prompt = prompt_encoding(tokenizer=tokenizer, prompt_text=prompt_text, args=args)

                ori_feedbacks = []
                for _ in range(args.num_paths):
                    total_sequence = get_output_sequence(model=model, encoded_prompt=encoded_prompt, args=args, tokenizer=tokenizer, prompt_text=prompt_text)
                    ori_feedbacks.append(total_sequence)
                keywords = ["New claim:", "new claim:","new Claim:","New Claim:", "modified claim:", "modified Claim:" "Modified Claim:", "Modified claim:","Edited Claim:","edited Claim:","edited claim:","Edited claim:", "updated claim:", "Updated claim:","updated Claim:","Updated Claim:"]
                pred_vote = improve_postprocess(engine=args.model_type, responses=ori_feedbacks, keywords=keywords, ori_sent=sent)

                if len(pred_vote['Entailed']) > len(pred_vote['Refuted']):
                    feedback_list.append(pred_vote['Entailed'][0].replace('\n','')) 
                else:
                    feedback_list.append(pred_vote['Refuted'][0].replace('\n',''))

            output_data[table_id] = feedback_list
            num += 1
            finetuned_model_names = ['GPT2', 'r2d2', 't5-base', 'plog-t5-large', 'loft']

        for finetuned in finetuned_model_names:
            if finetuned in args.finetuned_model_path:
                finetuned_model = finetuned
                break

    else:
        for cur_id, current in tqdm(data.items()):
            try:
                if num >= args.num_limit:
                    break

                # logical labels are only for LoTNLG
                table_id, prompt_text, logical_labels = get_prompt_from_table(dataset=args.dataset, current=current, prompt_template=prompt_template)

                if args.dataset != 'LoTNLG':
                    encoded_prompt = prompt_encoding(tokenizer=tokenizer, prompt_text=prompt_text, args=args)

                if args.dataset == 'LogicNLG':
                    sent_num = len(current["sentences"])
                    claim_list = []
                    for _ in range(sent_num):
                        total_sequence = get_output_sequence(model=model, encoded_prompt=encoded_prompt, args=args, tokenizer=tokenizer, prompt_text=prompt_text)
                        
                        keywords = ["Claim: "] 
                        include_keywords = False
                        for claim in total_sequence.split('\n'):
                            for keyword in keywords:
                                if keyword in claim:
                                    include_keywords = True
                                    claim = claim.replace(keyword, '').replace('</s>', '')
                                    claim_list.append(claim)
                        if include_keywords == False:
                            total_sequence = total_sequence.replace('</s>', '')
                            claim_list.append(total_sequence)

                    output_data[table_id] = claim_list

                elif args.dataset == 'FeTaQA' or args.dataset == 'F2WTQ':
                    total_sequence = get_output_sequence(model=model, encoded_prompt=encoded_prompt, args=args, tokenizer=tokenizer, prompt_text=prompt_text)
                        
                    keywords = ["Answer: ", "answer: "]
                    include_keyword = False    
                    for claim in total_sequence.split('\n'):
                        for keyword in keywords:
                            if keyword in claim:
                                include_keyword = True
                                feedback = claim.replace(keyword, '').strip()
                    if include_keyword == False:
                        feedback = total_sequence.replace('\n', '').strip()
                    feedback = feedback.replace('</s>', '')     #vicuna
                    output_data[table_id] = feedback

                elif args.dataset == 'LoTNLG':
                    sent_num = len(current["sentences"])
                    claim_list = []
                    for k in range(sent_num):
                        prompt_real_text = prompt_text.replace('{logical_label}', logical_labels[k])
                        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
                        if requires_preprocessing:
                            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
                            preprocessed_prompt_text = prepare_input(prompt_real_text)
                            encoded_prompt = tokenizer(preprocessed_prompt_text, return_tensors="pt").input_ids
                        else:
                            encoded_prompt = tokenizer(prompt_real_text, return_tensors="pt").input_ids

                        encoded_prompt = encoded_prompt.to("cuda")

                        output_sequence = model.generate(
                            input_ids=encoded_prompt,
                            max_length=args.length + len(encoded_prompt[0]),
                            temperature=args.temperature,
                            top_k=args.k,
                            top_p=args.p,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token is None else tokenizer.pad_token_id,
                        )[0]

                        # Decode text
                        text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)
                        # Remove all text after the stop token
                        text = text[text.find(prompt_real_text) + len(prompt_real_text):].split(args.stop_token)[0].strip()

                        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                        total_sequence = text

                        #print(total_sequence)
                        keywords = ["Claim: "] 
                        include_keywords = False
                        for claim in total_sequence.split('\n'):
                            for keyword in keywords:
                                if keyword in claim:
                                    include_keywords = True
                                    claim = claim.replace(keyword, '').replace('</s>', '')
                                    claim_list.append(claim)
                        if include_keywords == False:
                            total_sequence = total_sequence.replace('</s>', '')
                            claim_list.append(total_sequence)

                    output_data[table_id] = claim_list
                num += 1
            except:
                print(f'Warning! Current table id: {cur_id} cannot be processed! Skip this table!')
                output_data[table_id] = 'None'
                continue

    if args.direct_mode == 'CoT' and (args.dataset == 'FeTaQA' or args.dataset == 'F2WTQ'):
        output_data = FeTaQA_F2WTQ_CoT_clean(output_data)
    output_path = get_exact_output_path(output_path=args.output_path, engine=args.model_type, dataset=args.dataset, mode=args.mode, direct_mode=args.direct_mode, finetuned_model=finetuned_model)
    json.dump(output_data, open(output_path, "w"), indent=4)


if __name__ == "__main__":
    main()