deepspeed train_reranker.py \
    --model_name_or_path /h3cstore_nt/pc_embedding/mm3d/eedi/pretrain_model/Qwen2-Math-1.5B-Instruct \
    --max_length 1024 \
    --add_eos_token False \
    --train_data_path /h3cstore_nt/pc_embedding/mm3d/eedi/data/reranker_split/train_fold_1.json \
    --val_data_path /h3cstore_nt/pc_embedding/mm3d/eedi/data/reranker_split/val_fold_1.json \
    --output_dir /h3cstore_nt/pc_embedding/mm3d/eedi/output/reranker_fold_1 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target "["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]" \
    --gradient_checkpointing True \
    --eval_steps 0.2 \
    --eval_strategy "steps" \
    --bf16_full_eval True \
    --warmup_ratio 0.05 \
    --logging_steps 0.005 \
    --report_to "wandb" \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --lr_scheduler_type "cosine" \
    --run_name reranker_fold_1
