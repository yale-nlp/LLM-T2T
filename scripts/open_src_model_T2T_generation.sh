clear; export CUDA_VISIBLE_DEVICES=0,1; 

python ../src/open_src_model_T2T_generation.py \
    --model_type llama-7b \
    --model_name_or_path huggyllama/llama-7b \
    --mode direct \
    --direct CoT \
    --dataset FeTaQA \
    --output_path ../output/test_path \
    --num_limit 3

python ../src/open_src_model_T2T_generation.py \
    --model_type llama-7b \
    --model_name_or_path huggyllama/llama-7b \
    --mode improve \
    --dataset LogicNLG \
    --finetuned_model_path ../data/LogicNLG/original/100tables/GPT2_100tables.json \
    --output_path ../output/test_path \
    --num_limit 3