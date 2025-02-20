for mode in train val test; do
    if [ ! -d "data/semeval/$mode" ]; then
        mkdir -p results/semeval/$mode
    fi
done

export CUDA_VISIBLE_DEVICES=7

python3 src/run_prompt.py \
--data_dir data/semeval \
--output_dir results/semeval \
--model_type roberta \
--model_name_or_path roberta-large \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--max_seq_length 128 \
--warmup_steps 500 \
--learning_rate 3e-5 \
--learning_rate_for_new_token 1e-5 \
--num_train_epochs 6 \
--weight_decay 1e-2 \
--adam_epsilon 1e-6 \
--temps temp.txt
