import os
os.environ["HF_HOME"] = "/newssd/hf" 
from datasets import load_dataset

# 一度 Hugging Face から読み込み（キャッシュ済みなら高速）
from datasets import load_dataset_builder

builder = load_dataset_builder("Real-IAD/Real-IAD")
# print(builder.info)                # データセット全体のメタ情報
print(builder.info.features)       # 特徴量の型と内容
print(builder.info.splits)         # train/test/validationの構成
print(builder.info.description)    # 説明文
print(builder.info.license)        # ライセンス