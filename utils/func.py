import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    torch.backends.cudnn.deterministic = True  # 确保每次运行结果一致
    torch.backends.cudnn.benchmark = False  # 关闭基准测试，以确保可复现性
    
def print_matrix(matrix):
    # 获取矩阵的行数和列数
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    
    # 计算每列的最大宽度
    col_widths = []
    for j in range(cols):
        max_width = max(len(str(matrix[i][j])) for i in range(rows))
        col_widths.append(max_width)
    
    # 打印表头（如果需要）
    header = " | ".join(f"Col{j}" for j in range(cols))
    print(header)
    print("-" * len(header))
    
    # 打印每一行
    for row in matrix:
        row_str = " | ".join(f"{val:{col_widths[j]}}" for j, val in enumerate(row))
        print(row_str)
# def saveTensorAsImg(img,name="test",nrow=1):
#     vutils.save_image(img, f"./result/{name}.jpg",padding=0,normalize=True,nrow=nrow)

# def draw(img,cmap='gray'):
#     plt.imshow(img, cmap=cmap)  # 如果张量是灰度图像，可以使用 'gray' cmap
#     plt.axis('off')  # 不显示坐标轴
#     plt.save()




