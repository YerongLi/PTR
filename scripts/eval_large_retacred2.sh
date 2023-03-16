export CUDA_VISIBLE_DEVICES=0,1

python3 src/run_prompt.py \
--data_dir data/retacred \
--output_dir results/retacred2 \
--model_type roberta \
--model_name_or_path roberta-large \
--per_gpu_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--max_seq_length 512 \
--learning_rate_for_new_token 1e-5 \
--num_train_epochs 5 \
--adam_epsilon 1e-6 \
--temps temp.txt