# LLM_T2T
Logical Table-to-Text Generation by Large Language Models

## Table-to-Text Generation by GPT-series Models:
At first, modify the bash script [GPT_T2T_generation.sh](scripts/GPT_T2T_generation.sh). The description of all the arguments used in the script can be found in [GPT_T2T_generation.py](src/GPT_T2T_generation.py).

If you want to generate text directly by the LLM:

```
python ../src/GPT_T2T_generation.py \
    --api_org  \
    --api_key  \
    --engine gpt-3.5-turbo \
    --temperature 0.7 \
    --top_p 1.0 \
    --mode direct \
    --direct_mode two_shot_CoT \
    --dataset FeTaQA \
    --num_limit 3 \
    --output_path ../output/test_path/
```

If you want to improve the output of the finetuned small models by the LLM:

```
python ../src/GPT_T2T_generation.py \
    --api_org  \
    --api_key  \
    --engine gpt-3.5-turbo \
    --temperature 0.7 \
    --top_p 1.0 \
    --mode improve \
    --dataset LogicNLG \
    --finetuned_model_path ../output/LogicNLG/original/100tables/GPT2_100tables.json \
    --num_limit 3 \
    --num_paths 3 \
    --output_path ../output/test_path/
```
Then run the bash script:

```
sh GPT_T2T_generation.sh
```

## Table-to-Text Generation by Open-Source Models:
At first, modify the bash script [open_src_model_T2T_generation.sh](scripts/open_src_model_T2T_generation.sh). The description of all the arguments used in the script can be found in [open_src_model_T2T_generation.py](src/open_src_model_T2T_generation.py).

If you want to generate text directly by the LLM:

```
export CUDA_VISIBLE_DEVICES=0,1; 

python ../src/open_src_model_T2T_generation.py \
    --model_type llama-7b \
    --model_name_or_path huggyllama/llama-7b \
    --mode direct \
    --direct CoT \
    --dataset FeTaQA \
    --output_path ../output/test_path \
    --num_limit 3
```

If you want to improve the output of the finetuned small models by the LLM:

```
export CUDA_VISIBLE_DEVICES=0,1;

python ../src/open_src_model_T2T_generation.py \
    --model_type llama-7b \
    --model_name_or_path huggyllama/llama-7b \
    --mode improve \
    --dataset LogicNLG \
    --finetuned_model_path ../data/LogicNLG/original/100tables/GPT2_100tables.json \
    --output_path ../output/test_path \
    --num_limit 3
```
Then run the bash script:

```
sh open_src_model_T2T_generation.sh
```
