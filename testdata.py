#生成一个对角线为1+i，其他元素为0的69*69的复数矩阵矩
import torch
import numpy as np
import os

# 创建69x69的复数矩阵，对角线为1+i，其他元素为0
size = 69
matrix = np.zeros((size, size), dtype=np.complex128)
np.fill_diagonal(matrix, 1+1j)

# 重复5000遍
repeated_matrix = np.repeat(matrix[np.newaxis, :, :], 5000, axis=0)

# 分离实部和虚部，创建tensor
# 形状: (5000, 2, 69, 69) - 2个通道分别代表实部和虚部
real_part = repeated_matrix.real
imag_part = repeated_matrix.imag

# 堆叠实部和虚部
tensor = np.stack([real_part, imag_part], axis=1)

# 转换为PyTorch tensor
tensor = torch.from_numpy(tensor).float()

# 创建目录
os.makedirs('./data', exist_ok=True)

# 保存tensor
torch.save(tensor, './data/test_smos_D.pt')

# 输出tensor维度
print(f"Tensor维度: {tensor.shape}")
print(f"说明: (样本数, 通道数[实部/虚部], 高度, 宽度)")
print(f"已保存到: ./data/test_smos_D.pt")

# 验证数据
print(f"\n验证信息:")
print(f"第一个矩阵对角线元素 (实部): {tensor[0, 0, 0, 0].item()}")
print(f"第一个矩阵对角线元素 (虚部): {tensor[0, 1, 0, 0].item()}")
print(f"第一个矩阵非对角线元素 (实部): {tensor[0, 0, 0, 1].item()}")
print(f"第一个矩阵非对角线元素 (虚部): {tensor[0, 1, 0, 1].item()}")