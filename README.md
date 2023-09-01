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
			
### attack
CUDA_VISIBLE_DEVICES=0 python code/run_attack.py --data_dir ./data_codebert/ --output_dir ./model_codesearchnet/checkpoint-best-aver --encoder_name_or_path microsoft/codebert-base --pred_model_dir ./model_codesearchnet/checkpoint-last/
###