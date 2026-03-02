import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# 加载模型

model_path =f"GroupGrains/Results/DataNum5/x10_Net4x3-200-3(256)_Exp4e-04_0.9999/p[0.2-6.8-50000]xw[1e+05-1e+08-50000]/dem_epoch100000.pth"
state_dict = torch.load(model_path)

# 查找包含"latent_vectors"的键
latent_vectors_key = [key for key in state_dict.keys() if 'latent_vectors' in key]

if latent_vectors_key:
    latent_vectors = state_dict[latent_vectors_key[0]]
    print("Latent vectors shape:", latent_vectors.shape)
else:
    print("Latent vectors not found in the state dict")
    print("Available keys:", state_dict.keys())

# 获取要比较的向量
standard=0
vector_base = latent_vectors[standard].unsqueeze(0)  # 将向量变为[1, latent_dim]形状

# 计算余弦相似度
cos_similarities = []
for i in range(len(latent_vectors)):
    vector_i = latent_vectors[i].unsqueeze(0)
    # 计算余弦相似度
    similarity = F.cosine_similarity(vector_base, vector_i)
    cos_similarities.append(similarity.item())

# 打印结果
print(f"与latent_vectors[{standard}]的余弦相似度：")
for i in range(len(latent_vectors)):
    print(f"latent_vectors[{i}]的余弦相似度: {cos_similarities[i]:.4f}")

plt.figure(figsize=(10, 6))
indices = np.arange(len(latent_vectors))
plt.bar(indices, cos_similarities)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='g', linestyle='-', alpha=0.3)
plt.title(f'Cosine Similarity with latent_vectors[{standard}]')
plt.xlabel('Vector Index')
plt.ylabel('Cosine Similarity')
plt.xticks(indices)
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.3)
plt.savefig(f'cosine_similarities_{standard}.png')
plt.show()

