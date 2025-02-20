export CUDA_VISIBLE_DEVICES=0,1

python3 src/analyze.py \
--data_dir data/tacred \
--output_dir results/tacred \
--model_type roberta \
--model_name_or_path roberta-large \
--per_gpu_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--max_seq_length 512 \
--learning_rate_for_new_token 1e-5 \
--num_train_epochs 4 \
--temps temp.txt