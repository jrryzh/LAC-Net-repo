import imageio
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

depth_images = []

# depth_paths = sorted(glob.glob("/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/val/bin/depth/*.png"))
depth_paths = sorted(glob.glob("/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/val/tabletop/depth/*.png"))
# depth_paths = sorted(glob.glob("/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/disparity/*.png"))

for depth_path in depth_paths:
    depth_image = imageio.imread(depth_path)
    depth_images.append(depth_image)

# 将所有深度图像的像素值展平
depth_values = np.concatenate([img.flatten() for img in depth_images])

# 统计深度值的频次
histogram, bins = np.histogram(depth_values, bins=1000)  # 可以根据需要调整bins的数量

# 绘制直方图
plt.bar(bins[:-1], histogram, width=np.diff(bins), edgecolor='k')
plt.title('Depth Histogram')
plt.xlabel('Depth Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 保存直方图
plt.savefig('depth_histogram_uoais_tabletop.png')