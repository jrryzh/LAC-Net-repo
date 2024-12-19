import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms

from src.image_component import Resnet_Encoder, Refine_Module, RGBDLinearFusion
from utils.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
from utils.loss import CrossEntropyLoss

from utils.utils import torch_show_all_params, torch_init_model
from utils.utils import Config
from utils.evaluation import evaluation_image
from utils.loss import CrossEntropyLoss

class LAC_Net(nn.Module):
    def __init__(self, config, mode, logger=None, save_eval_dict={}):
        super(LAC_Net, self).__init__()
        self.config = config
        self.iteration = 0
        self.sample_iter = 0
        self.name = config.model_type

        self.root_path = config.path
        self.transformer_path = os.path.join(config.path, self.name)

        self.mode = mode
        self.save_eval_dict = save_eval_dict

        self.eps = 1e-6
        self.train_sample_iters = config.train_sample_iters
        
        # 初始化编码器和融合模块
        self.img_encoder = Resnet_Encoder().to(config.device)
        self.depth_encoder = Resnet_Encoder().to(config.device)
        self.refine_module = Refine_Module().to(config.device)

        # 初始化线性融合模块
        self.rgbd_linearfuse_256 = RGBDLinearFusion(256).to(config.device)
        self.rgbd_linearfuse_512 = RGBDLinearFusion(512).to(config.device)
        self.rgbd_linearfuse_1024 = RGBDLinearFusion(1024).to(config.device)
        self.rgbd_linearfuse_2048 = RGBDLinearFusion(2048).to(config.device)

        # 初始化损失函数
        self.refine_criterion = nn.BCELoss()
        self.criterion = CrossEntropyLoss(num_classes=config.vocab_size+1, device=config.device)

        # 初始化优化器
        optimizer_parameters = [
            {'params': self.img_encoder.parameters(), 'weight_decay': config.weight_decay},
            {'params': self.depth_encoder.parameters(), 'weight_decay': config.weight_decay},
            {'params': self.refine_module.parameters(), 'weight_decay': config.weight_decay},
            {'params': self.rgbd_linearfuse_256.parameters(), 'weight_decay': config.weight_decay},
            {'params': self.rgbd_linearfuse_512.parameters(), 'weight_decay': config.weight_decay},
            {'params': self.rgbd_linearfuse_1024.parameters(), 'weight_decay': config.weight_decay},
            {'params': self.rgbd_linearfuse_2048.parameters(), 'weight_decay': config.weight_decay},
        ]

        self.opt = AdamW(params=optimizer_parameters, lr=float(config.lr), betas=(config.beta1, config.beta2))
        self.sche = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_iters, num_training_steps=config.max_iters)

        try:
            self.rank = dist.get_rank()
        except:
            self.rank = 0

        self.gamma = self.gamma_func(mode=config.gamma_mode)
        self.mask_token_idx = config.vocab_size
        self.choice_temperature = 4.5
        self.Image_W = config.Image_W
        self.Image_H = config.Image_H
        self.patch_W = config.patch_W
        self.patch_H = config.patch_H

    def align_raw_size(self, full_mask, obj_position, vm_pad, meta):
        vm_np_crop = meta["vm_no_crop"].squeeze()
        H, W = vm_np_crop.shape[-2], vm_np_crop.shape[-1]
        bz, seq_len = full_mask.shape[:2]
        new_full_mask = torch.zeros((bz, seq_len, H, W)).to(torch.float32).cuda()
        if len(vm_pad.shape)==3:
            vm_pad = vm_pad[0]
            obj_position = obj_position[0]
        for b in range(bz):
            paddings = vm_pad[b]
            position = obj_position[b]
            new_fm = full_mask[
                b, :,
                :-int(paddings[0]) if int(paddings[0]) !=0 else None,
                :-int(paddings[1]) if int(paddings[1]) !=0 else None
            ]
            vx_min = int(position[0])
            vx_max = min(H, int(position[1])+1)
            vy_min = int(position[2])
            vy_max = min(W, int(position[3])+1)
            resize = transforms.Resize([vx_max-vx_min, vy_max-vy_min])
            try:
                new_fm = resize(new_fm)
                new_full_mask[b, :, vx_min:vx_max, vy_min:vy_max] = new_fm[0]
            except:
                new_fm = new_fm
        return new_full_mask

    def get_losses(self, meta):
        self.iteration += 1
        img_feat = self.img_encoder(meta['img_crop'].permute((0,3,1,2)).to(torch.float32))
        depth_feat = self.depth_encoder(meta['depth_crop'].permute((0,3,1,2)).to(torch.float32))
        # TODO: 待确定
        fusion_feat = []
        fusion_feat.append(self.rgbd_linearfuse_256(img_feat[0], depth_feat[0]))
        fusion_feat.append(self.rgbd_linearfuse_512(img_feat[1], depth_feat[1]))
        fusion_feat.append(self.rgbd_linearfuse_1024(img_feat[2], depth_feat[2]))
        fusion_feat.append(self.rgbd_linearfuse_2048(img_feat[3], depth_feat[3]))
        # import ipdb; ipdb.set_trace()

        # 修改： 将原来的transformer预测的coarse mask改为vm_crop_gt
        pred_fm_crop_old = meta["vm_crop_gt"]
        pred_vm_crop, pred_fm_crop = self.refine_module(fusion_feat, pred_fm_crop_old)

        pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256), mode="nearest")
        pred_vm_crop = torch.sigmoid(pred_vm_crop)
        loss_vm = self.refine_criterion(pred_vm_crop, meta['vm_crop_gt'])
        # pred_vm_crop = (pred_vm_crop>=0.5).to(torch.float32)

        pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode="nearest")
        pred_fm_crop = torch.sigmoid(pred_fm_crop)
        loss_fm = self.refine_criterion(pred_fm_crop, meta['fm_crop'])
        # pred_fm_crop = (pred_fm_crop>=0.5).to(torch.float32)
        logs = [
            ("loss_vm", loss_vm.item()),
            ("loss_fm", loss_fm.item()),
        ]
        return loss_vm+loss_fm, logs
    
    def loss_and_evaluation(self, pred_fm, meta, iter, mode, pred_vm=None):
        loss_eval = {}
        pred_fm = pred_fm.squeeze()
        counts = meta["counts"].reshape(-1).to(pred_fm.device)
        fm_no_crop = meta["fm_no_crop"].squeeze()
        vm_no_crop = meta["vm_no_crop"].squeeze()
        pred_vm = pred_vm.squeeze()
        # post-process
        pred_fm = (pred_fm > 0.5).to(torch.int64)
        pred_vm = (pred_vm > 0.5).to(torch.int64)
        
        iou, invisible_iou_, iou_count = evaluation_image((pred_fm > 0.5).to(torch.int64), fm_no_crop, counts, meta, self.save_eval_dict)
        loss_eval["iou"] = iou
        loss_eval["invisible_iou_"] = invisible_iou_
        loss_eval["occ_count"] = iou_count
        loss_eval["iou_count"] = torch.Tensor([pred_fm.shape[0]]).cuda()
        pred_fm_post = pred_fm + vm_no_crop
        
        pred_fm_post = (pred_fm_post>0.5).to(torch.int64)
        iou_post, invisible_iou_post, iou_count_post = evaluation_image(pred_fm_post, fm_no_crop, counts, meta, self.save_eval_dict)
        loss_eval["iou_post"] = iou_post
        loss_eval["invisible_iou_post"] = invisible_iou_post
        return loss_eval

    def backward(self, loss=None):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sche.step()

    @torch.no_grad()
    def calculate_metrics(self, meta_lst):
        pred_vm_lst, pred_fm_lst = [], []
        for meta in meta_lst:
            img_feat = self.img_encoder(meta['img_crop'].permute((0,3,1,2)).to(torch.float32))
            depth_feat = self.depth_encoder(meta['depth_crop'].permute((0,3,1,2)).to(torch.float32))
            #  TODO: 待确定
            fusion_feat = []
            fusion_feat.append(self.rgbd_linearfuse_256(img_feat[0], depth_feat[0]))
            fusion_feat.append(self.rgbd_linearfuse_512(img_feat[1], depth_feat[1]))
            fusion_feat.append(self.rgbd_linearfuse_1024(img_feat[2], depth_feat[2]))
            fusion_feat.append(self.rgbd_linearfuse_2048(img_feat[3], depth_feat[3]))
            import ipdb; ipdb.set_trace()

            # 修改： 将原来的transformer预测的coarse mask改为vm_crop_gt
            pred_fm_crop_old = meta["vm_crop_gt"]
            pred_vm_crop, pred_fm_crop = self.refine_module(fusion_feat, pred_fm_crop_old)

            pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256), mode="nearest")
            pred_vm_crop = torch.sigmoid(pred_vm_crop)
            loss_vm = self.refine_criterion(pred_vm_crop, meta['vm_crop_gt'])
            # pred_vm_crop = (pred_vm_crop>=0.5).to(torch.float32)

            pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode="nearest")
            pred_fm_crop = torch.sigmoid(pred_fm_crop)
            loss_fm = self.refine_criterion(pred_fm_crop, meta['fm_crop'])
            # pred_fm_crop = (pred_fm_crop>=0.5).to(torch.float32)

            pred_vm = self.align_raw_size(pred_vm_crop, meta['obj_position'], meta["vm_pad"], meta)
            pred_fm = self.align_raw_size(pred_fm_crop, meta['obj_position'], meta["vm_pad"], meta)
            
            pred_fm = pred_fm.squeeze()
            pred_vm = pred_vm.squeeze()
            pred_fm = (pred_fm > 0.5).to(torch.int64)
            pred_vm = (pred_vm > 0.5).to(torch.int64)
        
            pred_vm_lst.append(pred_vm)
            pred_fm_lst.append(pred_fm)
                 
        return pred_vm_lst, pred_fm_lst
    
    @torch.no_grad()
    def batch_predict(self, meta, iter, mode):
        '''
        :param x:[B,3,H,W] image
        :param c:[b,X,H,W] condition
        :param mask: [1,1,H,W] mask
        '''
        self.sample_iter += 1

        img_feat = self.img_encoder(meta['img_crop'].permute((0,3,1,2)).to(torch.float32))
        depth_feat = self.depth_encoder(meta['depth_crop'].permute((0,3,1,2)).to(torch.float32))

        fusion_feat = []
        fusion_feat.append(self.rgbd_linearfuse_256(img_feat[0], depth_feat[0]))
        fusion_feat.append(self.rgbd_linearfuse_512(img_feat[1], depth_feat[1]))
        fusion_feat.append(self.rgbd_linearfuse_1024(img_feat[2], depth_feat[2]))
        fusion_feat.append(self.rgbd_linearfuse_2048(img_feat[3], depth_feat[3]))

        pred_fm_crop_old = meta["vm_crop_gt"]
        pred_vm_crop, pred_fm_crop = self.refine_module(fusion_feat, pred_fm_crop_old)

        pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256), mode="nearest")
        pred_vm_crop = torch.sigmoid(pred_vm_crop)
        loss_vm = self.refine_criterion(pred_vm_crop, meta['vm_crop_gt'])

        pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode="nearest")
        pred_fm_crop = torch.sigmoid(pred_fm_crop)
        loss_fm = self.refine_criterion(pred_fm_crop, meta['fm_crop'])

        pred_vm = self.align_raw_size(pred_vm_crop, meta['obj_position'], meta["vm_pad"], meta)
        pred_fm = self.align_raw_size(pred_fm_crop, meta['obj_position'], meta["vm_pad"], meta)

        loss_eval = self.loss_and_evaluation(pred_fm, meta, iter, mode, pred_vm=pred_vm)
        loss_eval["loss_fm"] = loss_fm
        loss_eval["loss_vm"] = loss_vm
        
        return loss_eval

    def visualize(self, pred_vm, pred_fm, meta, mode, iteration):
        pred_fm = pred_fm.squeeze()
        pred_vm = pred_vm.squeeze()
        gt_vm = meta["vm_no_crop"].squeeze()
        gt_fm = meta["fm_no_crop"].squeeze()
        to_plot = torch.cat((pred_vm, pred_fm, gt_vm, gt_fm)).cpu().numpy()
        save_dir = os.path.join(self.root_path, '{}_samples'.format(mode))
        image_id, anno_id= meta["img_id"], meta["anno_id"]
        plt.imsave("{}/{}_{}_{}.png".format(save_dir, iteration, int(image_id.item()), int(anno_id.item())), to_plot)
    
    def create_inputs_tokens_normal(self, num, device):
        self.num_latent_size = self.config['resolution'] // self.config['patch_size']
        blank_tokens = torch.ones((num, self.num_latent_size ** 2), device=device)
        masked_tokens = self.mask_token_idx * blank_tokens

        return masked_tokens.to(torch.int64)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        elif mode == "log":
            return lambda r, total_unknown: - np.log2(r) / np.log2(total_unknown)
        else:
            raise NotImplementedError

    def load(self, is_test=False, prefix=None):
        if prefix is not None:
            transformer_path = self.transformer_path + prefix + '.pth'
        else:
            transformer_path = self.transformer_path + '_last.pth'
        if self.config.restore or is_test:
            if os.path.exists(transformer_path):
                print('Rank {} is loading {} Transformer...'.format(self.rank, transformer_path))
                data = torch.load(transformer_path, map_location="cpu")
                
                torch_init_model(self.img_encoder, transformer_path, 'img_encoder')
                torch_init_model(self.depth_encoder, transformer_path, 'depth_encoder')
                torch_init_model(self.rgbd_linearfuse_256, transformer_path, 'fusion_256')
                torch_init_model(self.rgbd_linearfuse_512, transformer_path, 'fusion_512')
                torch_init_model(self.rgbd_linearfuse_1024, transformer_path, 'fusion_1024')
                torch_init_model(self.rgbd_linearfuse_2048, transformer_path, 'fusion_2048')
                torch_init_model(self.refine_module, transformer_path, 'refine')

                if self.config.restore:
                    self.opt.load_state_dict(data['opt'])
                    # skip sche
                    from tqdm import tqdm
                    for _ in tqdm(range(data['iteration']), desc='recover sche...'):
                        self.sche.step()
                self.iteration = data['iteration']
                self.sample_iter = data['sample_iter']
            else:
                print(transformer_path, 'not Found')
                raise FileNotFoundError

    def save(self, prefix=None):
        if prefix is not None:
            save_path = self.transformer_path + "_{}.pth".format(prefix)
        else:
            save_path = self.transformer_path + ".pth"

        print('\nsaving {} {}...\n'.format(self.name, prefix))
        torch.save({
            'iteration': self.iteration,
            'sample_iter': self.sample_iter,
            'img_encoder': self.img_encoder.state_dict(),
            'depth_encoder': self.depth_encoder.state_dict(),
            'refine': self.refine_module.state_dict(),
            'fusion_256': self.rgbd_linearfuse_256.state_dict(),
            'fusion_512': self.rgbd_linearfuse_512.state_dict(),
            'fusion_1024': self.rgbd_linearfuse_1024.state_dict(),
            'fusion_2048': self.rgbd_linearfuse_2048.state_dict(),
            'opt': self.opt.state_dict(),
        }, save_path)
        
    @torch.no_grad()
    def inference(self, meta):
        '''
        :param x:[B,3,H,W] image
        :param c:[b,X,H,W] condition
        :param mask: [1,1,H,W] mask
        '''
        self.sample_iter += 1

        img_feat = self.img_encoder(meta['img_crop'].permute((0,3,1,2)).to(torch.float32))
        depth_feat = self.depth_encoder(meta['depth_crop'].permute((0,3,1,2)).to(torch.float32))
        #  TODO: 待确定
        fusion_feat = []
        fusion_feat.append(self.rgbd_linearfuse_256(img_feat[0], depth_feat[0]))
        fusion_feat.append(self.rgbd_linearfuse_512(img_feat[1], depth_feat[1]))
        fusion_feat.append(self.rgbd_linearfuse_1024(img_feat[2], depth_feat[2]))
        fusion_feat.append(self.rgbd_linearfuse_2048(img_feat[3], depth_feat[3]))

        # 修改： 将原来的transformer预测的coarse mask改为vm_crop_gt
        pred_fm_crop_old = meta["vm_crop_gt"]
        pred_vm_crop, pred_fm_crop = self.refine_module(fusion_feat, pred_fm_crop_old)

        pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256), mode="nearest")
        pred_vm_crop = torch.sigmoid(pred_vm_crop)

        pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode="nearest")
        pred_fm_crop = torch.sigmoid(pred_fm_crop)

        pred_vm = self.align_raw_size(pred_vm_crop, meta['obj_position'], meta["vm_pad"], meta)
        pred_fm = self.align_raw_size(pred_fm_crop, meta['obj_position'], meta["vm_pad"], meta)

        pred_fm = pred_fm.squeeze()
        pred_vm = pred_vm.squeeze()
        pred_fm = (pred_fm > 0.5).to(torch.int64)
        pred_vm = (pred_vm > 0.5).to(torch.int64)
        
        return pred_vm, pred_fm