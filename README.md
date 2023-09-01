# CodeNLAttack

## Setups
[![Python](https://img.shields.io/badge/python-3.7.15-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.13.0-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)
🤗 Transformers 4.25.1

## Natural Language-Code Search Task

- This task from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code)

## Train and Attack
```
# 1. fine-tune CodeBERT on NL-Code Search Task
python code/run_classifier.py
			--model_type roberta
			--do_predict
			--test_file test_webquery.json
			--max_seq_length 200
			--per_gpu_eval_batch_size 2
			--data_dir data
			--output_dir ./model_codesearchnet/checkpoint-best-aver
			--encoder_name_or_path microsoft/codebert-base
			--pred_model_dir ./model_codesearchnet/checkpoint-last
			--prediction_file ./evaluator/webquery_predictions.txt 
			
### 2. attack
CUDA_VISIBLE_DEVICES=0 python code/run_attack.py --data_dir ./data_codebert/ --output_dir ./model_codesearchnet/checkpoint-best-aver --encoder_name_or_path microsoft/codebert-base --pred_model_dir ./model_codesearchnet/checkpoint-last/
###