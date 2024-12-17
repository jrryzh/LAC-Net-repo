import os
import cv2
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import torch
from torch.utils.data import DataLoader
from data.dataloader_transformer import load_dataset
from utils.logger import setup_logger
from utils.utils import Config, to_cuda

from src.image_model_depth_linearfusion import LAC_Net

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--path', type=str, required=True, help='experiment path')
    parser.add_argument('--check_point_path', type=str, default="check_points")
    parser.add_argument('--dataset', type=str, default="UOAIS", help="select dataset")
    parser.add_argument('--batch', type=int, default=1)
    return parser.parse_args()

def setup_environment(args):
    """设置环境和配置"""
    args.path = os.path.join(args.check_point_path, args.path)
    os.makedirs(args.path, exist_ok=True)

    config_path = os.path.join(args.path, f'LAC_Net_{args.dataset}.yml')
    if not os.path.exists(config_path):
        copyfile(f'./configs/LAC_Net_{args.dataset}.yml', config_path)
    
    config = Config(config_path)
    config.path = args.path
    config.batch_size = args.batch
    config.dataset = args.dataset

    log_file = f'log-{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.txt'
    logger = setup_logger(os.path.join(args.path, 'logs'), logfile_name=log_file)

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available, this script supports only GPU execution.")

    config.device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    return config, logger

def evaluate_model(model, test_loader, config, logger):
    """评估模型性能"""
    iter = 0
    iou, iou_post, iou_count = 0, 0, 0
    invisible_iou_, invisible_iou_post, occ_count = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for items in tqdm(test_loader):
            items = to_cuda(items, config.device)
            loss_eval = model.batch_predict_maskgit(items, iter, 'test', T=3)
            iter += 1
            iou += loss_eval['iou']
            iou_post += loss_eval['iou_post']
            iou_count += loss_eval['iou_count']
            invisible_iou_ += loss_eval['invisible_iou_']
            invisible_iou_post += loss_eval['invisible_iou_post']
            occ_count += loss_eval['occ_count'].sum()

            logger.info(f'iter {iter-1}: iou: {loss_eval["iou"].item() / loss_eval["iou_count"].item()}, '
                        f'iou_post: {loss_eval["iou_post"].item() / loss_eval["iou_count"].item()}, '
                        f'occ: {loss_eval["invisible_iou_"].item() / (loss_eval["occ_count"].sum().item() + 1e-10)}, '
                        f'occ_post: {loss_eval["invisible_iou_post"].item() / (loss_eval["occ_count"].sum().item() + 1e-10)}')
            torch.cuda.empty_cache()

    logger.info(f'meanIoU: {iou.item() / iou_count.item()}')
    logger.info(f'meanIoU post-process: {iou_post.item() / iou_count.item()}')
    logger.info(f'meanIoU invisible: {invisible_iou_.item() / (occ_count.item() + 1e-10)}')
    logger.info(f'meanIoU invisible post-process: {invisible_iou_post.item() / (occ_count.item() + 1e-10)}')
    logger.info(f'iou_count: {iou_count}')
    logger.info(f'occ_count: {occ_count}')

if __name__ == '__main__':
    args = parse_arguments()
    config, logger = setup_environment(args)

    # 加载测试数据集
    test_dataset = load_dataset(config, args, "test")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        drop_last=True
    )

    # 初始化和加载模型
    model = LAC_Net(config, mode='test', logger=logger)
    model.load(is_test=True, prefix=config.stage2_iteration)
    model = model.to(config.device)

    # 评估模型
    evaluate_model(model, test_loader, config, logger)