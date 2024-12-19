import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import imageio
import random

# 文件夹路径
folder_path = "/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/val/tabletop/depth/"

# 读取文件夹中的所有深度图像
depth_images = []
for filename in random.sample(os.listdir(folder_path), 500):
# for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # 假设图像格式为png
        img_path = os.path.join(folder_path, filename)
        depth_image = imageio.imread(img_path)
        depth_images.append(depth_image)

# 随机选择20张深度图像进行分析
sampled_images = random.sample(depth_images, 20)

# 统计和绘制每张深度图像的直方图
for i, depth_image in enumerate(sampled_images):
    # 统计深度值的频次
    histogram, bins = np.histogram(depth_image.flatten(), bins=100)  # 可以根据需要调整bins的数量
    
    # 绘制直方图
    plt.figure()
    plt.bar(bins[:-1], histogram, width=np.diff(bins), edgecolor='k')
    plt.title(f'Depth Histogram - Image {i+1}')
    plt.xlabel('Depth Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # 保存直方图
    plt.savefig(f'depth_histogram_image_{i+1}.png')
