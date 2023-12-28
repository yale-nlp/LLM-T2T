# LLM-T2T
The data and code for the EMNLP 2023 industry-track paper [Investigating Table-to-Text Generation Capabilities of LLMs in Real-World Information Seeking Scenarios](https://arxiv.org/abs/2305.14987). This research investigates the table-to-text capabilities of different LLMs using four datasets within two real-world information seeking scenarios. It demonstrates that high-performing LLMs, such as GPT-4, can effectively serve as table-to-text generators, evaluators, and feedback generators.

## Environment Setup
The code is tested on the following environment:
- python 3.9.18
- CUDA 12.1
- run `pip install -r requirements.txt` to install all the required packages

## Data and Model Output
The dataset we used can be found in the `data` folder, and the model outputs are in the `output` folder.

## Table-to-Text Generation by GPT-series Models:
At first, modify the bash script [GPT_T2T_generation.sh](scripts/GPT_T2T_generation.sh). The description of all the arguments used in the script can be found in [GPT_T2T_generation.py](src/GPT_T2T_generation.py).

Commands to generate text directly by the LLM (RQ1):

```
export CUDA_VISIBLE_DEVICES=0,1; 
export PYTHONPATH=`pwd`;
python src/GPT_T2T_generation.py \
    --api_org  \
    --api_key  \
    --engine gpt-3.5-turbo \
    --temperature 0.7 \
    --top_p 1.0 \
    --mode direct \
    --direct_mode two_shot_CoT \
    --dataset FeTaQA \
    --output_path output/FeTaQA/
```

Commands to improve the output of the finetuned small models by the LLM (RQ3):

```
export CUDA_VISIBLE_DEVICES=0,1; 
export PYTHONPATH=`pwd`;
python src/GPT_T2T_generation.py \
    --api_org  \
    --api_key  \
    --engine gpt-3.5-turbo \
    --temperature 0.7 \
    --top_p 1.0 \
    --mode improve \
    --dataset LogicNLG \
    --finetuned_model_path output/LogicNLG/original/100tables/GPT2_100tables.json \
    --num_paths 3 \
    --output_path output/LogicNLG/
```

## Table-to-Text Generation by Open-Source Models:
At first, modify the bash script [open_src_model_T2T_generation.sh](scripts/open_src_model_T2T_generation.sh). The description of all the arguments used in the script can be found in [open_src_model_T2T_generation.py](src/open_src_model_T2T_generation.py).

Commands to generate text directly by the LLM (RQ1):

```
export CUDA_VISIBLE_DEVICES=0,1; 
export PYTHONPATH=`pwd`;
python src/open_src_model_T2T_generation.py \
    --model_type llama-7b \
    --model_name_or_path huggyllama/llama-7b \
    --mode direct \
    --direct CoT \
    --dataset FeTaQA \
    --output_path output/FeTaQA \
```

Commands to improve the output of the finetuned small models by the LLM (RQ3):

```
export CUDA_VISIBLE_DEVICES=0,1; 
export PYTHONPATH=`pwd`;
python src/open_src_model_T2T_generation.py \
    --model_type llama-7b \
    --model_name_or_path huggyllama/llama-7b \
    --mode improve \
    --dataset LogicNLG \
    --finetuned_model_path data/LogicNLG/original/100tables/GPT2_100tables.json \
    --output_path output/LogicNLG \
```

## Contact
For any issues or questions, kindly email us at: Yilun Zhao (yilun.zhao@yale.edu), Haowei Zhang (haowei.zhang@tum.de) or Shengyun Si (shengyun.si@tum.de).

## Citation
```
@inproceedings{zhao-etal-2023-investigating,
    title = "Investigating Table-to-Text Generation Capabilities of Large Language Models in Real-World Information Seeking Scenarios",
    author = "Zhao, Yilun  and
      Zhang, Haowei  and
      Si, Shengyun  and
      Nan, Linyong  and
      Tang, Xiangru  and
      Cohan, Arman",
    editor = "Wang, Mingxuan  and
      Zitouni, Imed",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-industry.17",
    doi = "10.18653/v1/2023.emnlp-industry.17",
    pages = "160--175",
}
```
