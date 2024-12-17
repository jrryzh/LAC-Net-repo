import os
import cv2
import time
import random
import argparse
import numpy as np
import torch
from shutil import copyfile
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataloader_transformer import load_dataset
from utils.logger import setup_logger
from utils.utils import Config, Progbar, to_cuda

from src.image_model_depth_linearfusion import LAC_Net

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path', type=str, required=True, help='experiment path')
    parser.add_argument('--check_point_path', type=str, default="check_points")
    parser.add_argument('--dataset', type=str, default="UOAIS", help="select dataset")
    parser.add_argument('--batch', type=int, default=1)
    return parser.parse_args()

def setup_environment(args):
    torch.cuda.set_device(0)
    args.path = os.path.join(args.check_point_path, args.path)
    os.makedirs(args.path, exist_ok=True)
    config_path = os.path.join(args.path, f'LAC_Net_{args.dataset}.yml')
    if not os.path.exists(config_path):
        copyfile(f'./configs/LAC_Net_{args.dataset}.yml', config_path)
    config = Config(config_path)
    config.path = args.path
    config.batch_size = args.batch
    config.dataset = args.dataset
    return config

def setup_logger_and_device(config):
    log_file = f'log-{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.txt'
    logger = setup_logger(os.path.join(config.path, 'logs'), logfile_name=log_file)
    os.makedirs(os.path.join(config.path, 'val_samples'), exist_ok=True)
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        config.device = torch.device("cpu")
    return logger

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(args, config, logger):
    
    model = LAC_Net(config, mode='train', logger=logger)
    model.load(is_test=False, prefix=config.stage2_iteration)
    model.to(config.device)
    train_dataset, test_dataset = load_dataset(config, args, "train")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.train_num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.test_num_workers,
        drop_last=False,
    )
    sample_iterator = test_dataset.create_iterator(config.sample_size)
    steps_per_epoch = len(train_dataset) // config.batch_size
    iteration = model.iteration
    sample_iter = model.sample_iter
    epoch = model.iteration // steps_per_epoch
    logger.info(f'Start from epoch:{epoch}, iteration:{iteration}')
    writer = SummaryWriter(args.path)
    model.train()
    keep_training = True
    best_score = {}
    if config.lm_rate > 0:
        lm_size = int(config.batch_size * config.lm_rate)
        mc = [1] * lm_size + [0] * (config.batch_size - lm_size)
    else:
        mc = None

    while keep_training:
        epoch += 1
        stateful_metrics = ['epoch', 'iter', 'lr']
        progbar = Progbar(len(train_dataset), max_iters=steps_per_epoch,
                        width=20, stateful_metrics=stateful_metrics)
        
        for items in train_loader:
            model.train()
            items = to_cuda(items, config.device)
            items['mc'] = None
            counts = items['counts'].squeeze().bool().sum()
            if counts == 0:
                continue
            r_loss, logs = model.get_losses(items)
            model.backward(r_loss)
            torch.cuda.empty_cache()
            iteration = model.iteration
            sample_iter = model.sample_iter

            writer.add_scalar("loss/r_loss", r_loss, model.iteration)
            logs = [("epoch", epoch), ("iter", iteration), ('lr', model.sche.get_lr()[0])] + logs
            progbar.add(config.batch_size, values=logs)

            if iteration % config.val_vis_iters == 0:
                validate_and_visualize(args, config, model, sample_iterator)

            if iteration % config.log_iters == 0:
                logger.debug(str(logs))

            if iteration % config.save_iters == 0:
                model.save(prefix=f'{iteration}')

            if iteration >= config.max_iters:
                keep_training = False
                break

def validate_and_visualize(args, config, model, sample_iterator):
    model.eval()
    with torch.no_grad():
        items = next(sample_iterator)
        items = to_cuda(items, config.device)
        
        img_id = items["img_id"]
        anno_id = items["anno_id"]
        sample_iter = model.sample_iter
        loss_eval = model.batch_predict_maskgit(items, sample_iter, 'val')
        iou = loss_eval['iou'].item() / (loss_eval['iou_count'].item() + 1e-7)
        invisible_iou_ = loss_eval['invisible_iou_'].item()
        print(
            "img_id: ", int(anno_id[0].cpu().detach().numpy()), " - ",
            "anno_id: ", int(anno_id[0].cpu().detach().numpy()), " - ",
            "Iou: ", iou, " - ",
            "in-Iou", invisible_iou_, 
        )

if __name__ == '__main__':
    args = parse_arguments()
    config = setup_environment(args)
    logger = setup_logger_and_device(config)
    set_random_seed(config.seed)
    train_model(args, config, logger)