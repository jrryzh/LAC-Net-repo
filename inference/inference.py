import os
import cv2
import argparse
import numpy as np
import torch
from utils.utils import Config, to_cuda
import re
from PIL import Image
import imageio
from src.image_model_depth_linearfusion import C2F_Seg
from infdataset import inf_dataloader

def add_mask(mask, img, color1, color_mask=np.array([0, 0, 255]), line_width=1):
    # 将mask转换为布尔类型
    mask = mask.astype(np.bool)
    # 查找轮廓
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    res = cv2.drawContours(img.copy(), contours, -1, color1, line_width)
    # 将mask区域的颜色进行混合
    res[mask] = res[mask] * 0.7 + color_mask * 0.3
    return res

def load_model(config, device):
    # 初始化并加载模型
    model = C2F_Seg(config, mode='test')
    model.load(is_test=True, prefix=config.stage2_iteration)
    model = model.to(device)
    model.eval()
    return model

def process_files(directory_path, output_path, model, device):
    # 编译正则表达式模式
    pattern = re.compile(r'_visible_mask_\d+\.jpg$')
    visible_mask_files = []
    filename_prefixes = []

    # 遍历目录中的文件
    for filename in os.listdir(directory_path):
        if pattern.search(filename):
            visible_mask_files.append(filename)
            prefix = re.split(r'_visible_mask_\d+\.jpg', filename)[0]
            filename_prefixes.append(prefix)

    # 处理每个文件
    for vm_file, filename in zip(visible_mask_files, filename_prefixes):
        img_path = os.path.join("/cpfs/2926428ee2463e44/user/zjy/code_repo/c2f-seg/inference/one_head/color", filename+".png")
        depth_path = os.path.join("/cpfs/2926428ee2463e44/user/zjy/code_repo/c2f-seg/inference/one_head/depth", filename+".png")
        vm_path = os.path.join(directory_path, vm_file)

        # 加载数据
        items = inf_dataloader(img_path, depth_path, vm_path)
        items = to_cuda(items, device)

        # 模型推理
        _, pred_fm = model.inference(items)
        pred_fm_np = pred_fm.squeeze().cpu().numpy()

        # 转换数据类型为 uint8
        pred_fm_np_uint8 = (pred_fm_np * 255).astype(np.uint8)

        # 保存预测结果
        Image.fromarray(pred_fm_np_uint8).save(os.path.join(output_path, filename+"_lacnet_mask.png"))

        # 读取原始图像并添加mask
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        color1 = [255, 0, 0]
        color2 = np.array(color1) + 35
        masked_img = add_mask(pred_fm_np, img, color1, color2, 2)
        imageio.imwrite(os.path.join(output_path, filename+"_lacnet_masked.png"), masked_img)

        print(f"finished on {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--check_point_path', type=str, default="check_points")
    parser.add_argument('--dataset', type=str, default="UOAIS", help="select dataset")
    parser.add_argument('--data_type', type=str, default="image", help="select image or video model")
    parser.add_argument('--batch', type=int, default=1)

    args = parser.parse_args()

    # 构建模型路径
    args.path = os.path.join(args.check_point_path, args.path)
    os.makedirs(args.path, exist_ok=True)

    # 加载配置文件
    config_path = os.path.join(args.path, 'c2f_seg_{}.yml'.format(args.dataset))
    config = Config(config_path)

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available, this script supports only GPU execution.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)

    # 加载模型
    model = load_model(config, device)

    # 指定输入和输出路径
    directory_path = '/cpfs/2926428ee2463e44/user/zjy/code_repo/c2f-seg/inference/grounded_sam_output/one_head'
    output_path = '/cpfs/2926428ee2463e44/user/zjy/code_repo/c2f-seg/inference/lacnet_output/one_head'

    # 处理文件
    process_files(directory_path, output_path, model, device)

if __name__ == '__main__':
    main()