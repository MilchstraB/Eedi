python inference.py \
    --model_name_or_path /h3cstore_nt/pc_embedding/mm3d/eedi/pretrain_model/bge-small-en-v1.5 \
    --model_max_length 1024 \
    --half_precision True \
    --train_data_path /h3cstore_nt/pc_embedding/mm3d/eedi/data/train_after_process.csv \
    --misconception_mapping /h3cstore_nt/pc_embedding/mm3d/eedi/data/misconception_mapping.csv \
    --batch_size 8
