import json
import pandas as pd
from sklearn.model_selection import KFold

# 加载 JSON 文件为 Pandas DataFrame
json_objects = []
with open('/h3cstore_nt/pc_embedding/mm3d/eedi/data/train_hn.jsonl', 'r') as f:
    for line in f:
        json_objects.append(json.loads(line))

df = pd.DataFrame(json_objects)

# 初始化 KFold（5折交叉验证）
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储每一折的训练集和验证集
folds = []

# 划分数据集
for fold_idx, (train_index, val_index) in enumerate(kf.split(df)):
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]
    
    # 保存为字典格式
    train_data = train_df.to_dict(orient='records')
    val_data = val_df.to_dict(orient='records')
    
    # 将每折的训练集和验证集存储到 folds 列表中
    folds.append({
        'fold': fold_idx + 1,
        'train': train_data,
        'val': val_data
    })

    # 输出折的信息
    print(f"Fold {fold_idx + 1}:")
    print(f"  - Train size: {len(train_data)}")
    print(f"  - Val size: {len(val_data)}")

# 将每一折的数据保存到 JSON 文件中
for fold in folds:
    fold_num = fold['fold']
    
    # 保存训练集
    with open(f'/h3cstore_nt/pc_embedding/mm3d/eedi/data/reranker_split/train_fold_{fold_num}.json', 'w') as f:
        json.dump(fold['train'], f, indent=4)
    
    # 保存验证集
    with open(f'/h3cstore_nt/pc_embedding/mm3d/eedi/data/reranker_split/val_fold_{fold_num}.json', 'w') as f:
        json.dump(fold['val'], f, indent=4)

print("5折交叉验证数据集划分完成并保存至文件。")
