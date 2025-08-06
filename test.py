# 快速查看版本
from datasets import load_dataset
import json

# 只获取数据集大小，不下载全部数据
train_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train", streaming=True)
test_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test", streaming=True)

print("正在计算数据集大小...")

# # 对于streaming dataset，需要遍历来计算大小
# train_count = sum(1 for _ in train_dataset)
# test_count = sum(1 for _ in test_dataset)

# print(f"训练集: {train_count:,} 条数据")
# print(f"测试集: {test_count:,} 条数据")
# print(f"总计: {train_count + test_count:,} 条数据")

# # 基于你的配置计算
# batch_size = 32
# batches_per_epoch = train_count // batch_size
# print(f"\n每个epoch批次数: {batches_per_epoch}")
# print(f"100个epochs总处理量: {train_count * 100:,} 个样本")

# 打印一条数据
print("\n=== 数据样本 ===")
train_dataset_sample = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train", streaming=True)
sample = next(iter(train_dataset_sample))
print("训练集中的一条数据:")
for key, value in sample.items():
    print(f"{key}: {value}")

# 保存一条数据到JSON文件
output_file = "sample_data.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)

print(f"\n数据已保存到: {output_file}")