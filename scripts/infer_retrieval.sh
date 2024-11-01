python infer_retrieval.py \
    --model_name_or_path "<your_model_path>" \
    --lora_dir "<lora_weight_path>" \
    --val_data_path data/retrieval/val_fold_1.json \
    --misconception_mapping /path/to/output \
    --max_length 1024 \
    --add_eos_token True \
    --eval_batch_size 8 \
    --half_precision True \
    --top_k_for_retrieval 25 \
    --sentence_pooling_method "last"