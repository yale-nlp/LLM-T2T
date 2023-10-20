
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
