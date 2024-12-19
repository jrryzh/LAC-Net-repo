import os
import cv2
import random
import numpy as np
import cvbase as cvb
from PIL import Image
from skimage import transform
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pycocotools.mask as mask_utils
import cv2
import numpy as np
import torch
import imageio
import pyfastnoisesimd as fns

def perlin_noise(frequency, width, height):

    noise = fns.Noise()
    noise.NoiseType = 2 # perlin noise
    noise.frequency = frequency
    result = noise.genAsGrid(shape=[height, width], start=[0,0])
    return result

def PerlinDistortion(image, width, height):
    """
    """
    # sample distortion parameters from noise vector
    fx = np.random.uniform(0.0001, 0.1)
    fy = np.random.uniform(0.0001, 0.1)
    fz = np.random.uniform(0.01, 0.1)
    wxy = np.random.uniform(0, 10)
    wz = np.random.uniform(0, 0.005)
    cnd_x = wxy * perlin_noise(fx, width, height)
    cnd_y = wxy * perlin_noise(fy, width, height)
    cnd_z = wz * perlin_noise(fz, width, height)

    cnd_h = np.array(list(range(height)))
    cnd_h = np.expand_dims(cnd_h, -1)
    cnd_h = np.repeat(cnd_h, width, -1)
    cnd_w = np.array(list(range(width)))
    cnd_w = np.expand_dims(cnd_w, 0)
    cnd_w = np.repeat(cnd_w, height, 0)

    noise_cnd_h = np.int16(cnd_h + cnd_x)
    noise_cnd_h = np.clip(noise_cnd_h, 0, (height - 1))
    noise_cnd_w = np.int16(cnd_w + cnd_y)
    noise_cnd_w = np.clip(noise_cnd_w, 0, (width - 1))

    new_img = image[(noise_cnd_h, noise_cnd_w)]
    new_img = new_img = new_img + cnd_z
    return new_img.astype(np.float32)

def normalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ normalize the input depth (mm) and return depth image (0 ~ 255)
    Args:
        depth ([np.float]): depth array [H, W] (mm) 
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.

    Returns:
        [np.uint8]: normalized depth array [H, W, 3] (0 ~ 255)
    """
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))
    return depth

def unnormalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ unnormalize the input depth (0 ~ 255) and return depth image (mm)
    Args:
        depth([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.
    Returns:
        [np.float]: depth array [H, W] (mm) 
    """
    depth = np.float32(depth) / 255
    depth = depth * (max_val - min_val) + min_val
    return depth


def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    """ inpaint the input depth where the value is equal to zero

    Args:
        depth ([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        factor (int, optional): resize factor in depth inpainting. Defaults to 4.
        kernel_size (int, optional): kernel size in depth inpainting. Defaults to 5.

    Returns:
        [np.uint8]: inpainted depth array [H, W, 3] (0 ~ 255)
    """
    
    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W//factor, H//factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth


def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor


# class UOAIS_VQ_mask(torch.utils.data.Dataset):
#     def __init__(self, config, mode):
#         super(UOAIS_VQ_mask, self).__init__()
#         self.config = config
#         self.mode = mode

#         root_path = config.root_path
#         # temporal
#         train_img_path = "/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/train/bin/color"
#         test_img_path = "/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/val/bin/color"
#         self.image_list = []
#         if mode=="train":
#             # train_label = cvb.load(os.path.join(root_path,"COCOA_annotations_detectron/COCO_amodal_train2014_with_classes.json"))
#             train_label = cvb.load("/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/annotations/coco_anns_uoais_sim_train.json")
#             self.anns_dict_list = train_label["annotations"]
#             self.img_dict_list = train_label["images"]
#             # flist = config.data_flist.train
#         elif mode=="test":
#             # val_label = cvb.load(os.path.join(root_path,"COCOA_annotations_detectron/COCO_amodal_val2014_with_classes.json"))
#             val_label = cvb.load("/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/annotations/coco_anns_uoais_sim_val.json")
#             self.anns_dict_list = val_label["annotations"]
#             self.img_dict_list = val_label["images"]
#             # flist = config.data_flist.test
            
#         # self.image_list = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
#         # self.data_list = list(self.anns_dict)
#         self.data_list = self.anns_dict_list
#         self.dtype = torch.float32
#         self.enlarge_coef = 2
#         self.patch_h = 256
#         self.patch_w = 256
#         self.device = "cpu"

#     def __len__(self):
#         return len(self.img_dict_list)

#     def __getitem__(self, index):
#         return self.load_item(index)
        
#     def load_item(self, index):
#         if self.mode=="train":
#             self.enlarge_coef = random.uniform(1.5, 3)
        
#         if self.mode=="train":
#             img_root_path = "/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/train/"
#         else:
#             img_root_path = "/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/val/"
        
#         img = cv2.imread(os.path.join(img_root_path, self.img_dict_list[index]["file_name"]), cv2.IMREAD_COLOR)
#         img = img[...,::-1]
#         height, width, _ = img.shape
        
#         ann = self.anns_dict_list[index]

#         full_mask = ann["segmentation"]
#         fm_no_crop = mask_utils.decode(full_mask)

#         y_min, x_min, w, h = ann["bbox"]
#         y_max, x_max = y_min + w, x_min + h
#         y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max) 

#         x_center = (x_min + x_max) // 2
#         y_center = (y_min + y_max) // 2
#         x_len = int((x_max - x_min) * self.enlarge_coef)
#         y_len = int((y_max - y_min) * self.enlarge_coef)
#         x_min = max(0, x_center - x_len // 2)
#         x_max = min(height, x_center + x_len // 2)
#         y_min = max(0, y_center - y_len // 2)
#         y_max = min(width, y_center + y_len // 2)
        
#         fm_crop = fm_no_crop[x_min:x_max+1, y_min:y_max+1].astype(bool)
#         h, w = fm_crop.shape[:2]
#         m = transform.rescale(fm_crop, (self.patch_h/h, self.patch_w/w))
#         cur_h, cur_w = m.shape[:2]
#         to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
#         m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]    
#         fm_crop = m[np.newaxis, ...]

#         fm_crop = torch.from_numpy(fm_crop).to(self.dtype).to(self.device)

#         meta = {
#             "mask_crop": fm_crop,
#         }

#         return meta

#     @staticmethod
#     def collate_fn(batch):
#         keys = batch[0].keys()
#         res = {}
#         for k in keys:
#             temp_ = []
#             for b in batch:
#                 if b[k] is not None:
#                     temp_.append(b[k])
#             if len(temp_) > 0:
#                 res[k] = default_collate(temp_)
#             else:
#                 res[k] = None

#         return res

#     def create_iterator(self, batch_size):
#         while True:
#             sample_loader = DataLoader(
#                 dataset=self,
#                 batch_size=batch_size,
#                 drop_last=True,
#                 collate_fn=self.collate_fn
#             )

#             for item in sample_loader:
#                 yield item

#     def make_json_dict(self, imgs, anns):
#         imgs_dict = {}
#         anns_dict = {}
#         for ann in anns:
#             image_id = ann["image_id"]
#             if not image_id in anns_dict:
#                 anns_dict[image_id] = []
#                 anns_dict[image_id].append(ann)
#             else:
#                 anns_dict[image_id].append(ann)
        
#         for img in imgs:
#             image_id = img['id']
#             imgs_dict[image_id] = img['file_name']

#         return imgs_dict, anns_dict

#     def polys_to_mask(self, polygons, height, width):
#         rles = mask_utils.frPyObjects(polygons, height, width)
#         rle = mask_utils.merge(rles)
#         mask = mask_utils.decode(rle)
#         return mask

class Fusion_UOAIS_ALLVM(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(Fusion_UOAIS_ALLVM, self).__init__()
        self.config = config
        self.mode = mode
        # self.data_info = pickle.load(open(os.path.join("/home/ubuntu/data/Fusion_cocoa","retrain_{}.pkl".format(self.mode)),"rb"))
        # self.label_info = np.genfromtxt(os.path.join("/home/ubuntu/data/Fusion_cocoa", "{}_lable.txt".format(self.mode)), dtype=np.str, encoding='utf-8')
        if mode=="train":
            train_label = cvb.load("/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/annotations/coco_anns_uoais_sim_train.json")
            self.anns_dict_list = train_label["annotations"]
            self.img_dict_list = train_label["images"]
        elif mode=="test":
            val_label = cvb.load("/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/annotations/coco_anns_uoais_sim_val.json")
            self.anns_dict_list = val_label["annotations"]
            self.img_dict_list = val_label["images"]
        if self.mode=="train":
            self.img_root_path = "/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/train/"
        elif self.mode=="test":
            self.img_root_path = "/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/val/"
            
        # 修改：添加事先计算好的vm list
        self.vm_lst = cvb.load("/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/annotations/best_vm_matches.json")

        # self.uoais_vm_annotation_list = cvb.load("/home/zhangjinyu/uoais_new/predvms/pred_vis_masks_5k.json")
        self.dtype = torch.float32
        self.enlarge_coef = 2
        self.patch_h = 256
        self.patch_w = 256
        self.device = "cpu"
        
        self.img_size = (640, 480)
        self.depth_min, self.depth_max = [2500, 15000]

        
    def __len__(self):
        return len(self.anns_dict_list)

    def __getitem__(self, index):
        return self.load_item(index)
        
    def load_item(self, index):
        # anno_id, img_path = self.label_info[index].split(",")
        img = cv2.imread(os.path.join(self.img_root_path, self.img_dict_list[index]["file_name"]), cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        # depth = imageio.imread(os.path.join(self.img_root_path, self.img_dict_list[index]["depth_file_name"]))
        # if self.img_dict_list[index]["depth_file_name"].startswith("bin"):
        #     depth = normalize_depth(depth, min_val=2000.0, max_val=13000.0)
        # elif self.img_dict_list[index]["depth_file_name"].startswith("tabletop"):
        #     peak1, peak2, mean = 5000.0, 20000.0, np.mean(depth) 
        #     if abs(peak1 - mean) < abs(peak2 - mean):   # 属于左峰
        #         depth = normalize_depth(depth, min_val=0.0, max_val=25000.0)
        #     else:                                       # 属于右峰
        #         depth = normalize_depth(depth, min_val=0.0, max_val=25000.0)
        # else:
        #     raise Exception("wrong name in depth file")
        # depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        # depth = inpaint_depth(depth)
        # 修改：处理depth
        depth = imageio.imread(os.path.join(self.img_root_path, self.img_dict_list[index]["depth_file_name"])).astype(np.float32)
        if self.mode == "train":
            # if random.random() > 0.5:
            depth = PerlinDistortion(depth, *self.img_size)
            # depth = PerlinDistortion(depth, depth.shape[1], depth.shape[0])
        depth[depth > self.depth_max] = self.depth_max
        depth[depth < self.depth_min] = self.depth_min
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min) * 255
        depth = np.expand_dims(depth, -1)
        depth = np.uint8(np.repeat(depth, 3, -1))

        ann = self.anns_dict_list[index]
        anno_id = ann["id"]
        img_id = ann["image_id"]
        category_id = ann["category_id"]

        full_mask = ann["segmentation"]
        fm_no_crop = mask_utils.decode(full_mask)[...,np.newaxis]

        fm_no_crop_gt = fm_no_crop

        # 修改： 将这个地方的 gt vm 换成计算后的vm
        # visible_mask = ann["visible_mask"]
        visible_mask = self.vm_lst[index]
        # 如果 vm 为 None 则仍用 gt vm
        if visible_mask is None:
            visible_mask = ann["visible_mask"]
        vm_no_crop = mask_utils.decode(visible_mask)[...,np.newaxis]

        vm_no_crop_gt = vm_no_crop

        y_min, x_min, w, h = ann["bbox"]
        y_max, x_max = y_min + w, x_min + h
        y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max) 

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        x_len = int((x_max - x_min) * self.enlarge_coef)
        y_len = int((y_max - y_min) * self.enlarge_coef)
        x_min = max(0, x_center - x_len // 2)
        x_max = min(height, x_center + x_len // 2)
        y_min = max(0, y_center - y_len // 2)
        y_max = min(width, y_center + y_len // 2)
        
        fm_crop = fm_no_crop[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)
        vm_crop = vm_no_crop[x_min:x_max+1, y_min:y_max+1, 0].astype(bool)
        img_crop = img[x_min:x_max+1, y_min:y_max+1]
        depth_crop = depth[x_min:x_max+1, y_min:y_max+1]

        h, w = vm_crop.shape[:2]
        m = transform.rescale(vm_crop, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
        vm_crop = m[np.newaxis, ...]

        img_ = transform.rescale(img_crop, (self.patch_h/h, self.patch_w/w, 1))
        cur_h, cur_w = img_.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
        img_ = np.pad(img_, to_pad)[:self.patch_h, :self.patch_w, :3]
        img_crop = img_

        # 修改：添加depth
        depth_ = transform.rescale(depth_crop, (self.patch_h/h, self.patch_w/w, 1))
        cur_h, cur_w = depth_.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
        depth_ = np.pad(depth_, to_pad)[:self.patch_h, :self.patch_w, :3]
        depth_crop = depth_

        m = transform.rescale(fm_crop, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]    
        fm_crop = m[np.newaxis, ...]

        obj_position = np.array([x_min, x_max, y_min, y_max])
        vm_pad = np.array([max(self.patch_h-cur_h, 0), max(self.patch_w-cur_w, 0)])
        vm_scale = np.array([self.patch_h/h, self.patch_w/w])

        # full_pad = ((0, max(375-height, 0)), (0, max(1242-width, 0)))
        # vm_no_crop = np.pad(vm_no_crop, full_pad)[:375, :1242]
        # fm_no_crop = np.pad(fm_no_crop, full_pad)[:375, :1242]
        vm_no_crop = vm_no_crop[np.newaxis, ...]
        fm_no_crop = fm_no_crop[np.newaxis, ...]

        loss_mask = fm_no_crop-vm_no_crop
        loss_mask[loss_mask==255]=0
        loss_mask = 1-loss_mask.astype(bool)
        # data augmentation
        vm_crop_aug = self.data_augmentation(vm_crop[0])[np.newaxis, ...]
        
        counts = np.array([1])
        counts = torch.from_numpy(counts).to(self.dtype).to(self.device)

        obj_position = torch.from_numpy(obj_position).to(self.dtype).to(self.device)
        vm_pad = torch.from_numpy(vm_pad).to(self.dtype).to(self.device)
        vm_scale = torch.from_numpy(vm_scale).to(self.dtype).to(self.device)

        fm_crop = torch.from_numpy(fm_crop).to(self.dtype).to(self.device)
        fm_no_crop = torch.from_numpy(np.array(fm_no_crop)).to(self.dtype).to(self.device)
        vm_crop = torch.from_numpy(vm_crop).to(self.dtype).to(self.device)
        vm_crop_aug = torch.from_numpy(vm_crop_aug).to(self.dtype).to(self.device)
        img_crop = torch.from_numpy(img_crop).to(self.dtype).to(self.device)
        img = torch.from_numpy(img).to(self.dtype).to(self.device)
        # 修改： 添加depth
        depth_crop = torch.from_numpy(depth_crop).to(self.dtype).to(self.device)
        depth = torch.from_numpy(depth).to(self.dtype).to(self.device)
        vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(self.dtype).to(self.device)
        
        loss_mask = torch.from_numpy(np.array(loss_mask)).to(self.dtype).to(self.device)
    
        img_id = torch.from_numpy(np.array(img_id)).to(self.dtype).to(self.device)
        anno_id = torch.from_numpy(np.array(anno_id)).to(self.dtype).to(self.device)
        category_id = torch.from_numpy(np.array(category_id)).to(self.dtype).to(self.device)
        if self.mode=="train":
            meta = {
                # "vm_no_crop": vm_no_crop,
                "vm_crop": vm_crop_aug,
                "vm_crop_gt": vm_crop,
                # "fm_no_crop": fm_no_crop,
                "fm_crop": fm_crop,
                "img_crop": img_crop,
                "depth_crop": depth_crop,    # 修改： 添加depth            
                # "loss_mask": loss_mask,
                "obj_position": obj_position,
                "vm_pad": vm_pad,
                "vm_scale": vm_scale,
                "counts": counts,
                "img_id": img_id,
                "anno_id": anno_id,
                "category_id": category_id,
                # for vq
                # "mask_crop": fm_crop
                # "img_no_crop": img
                "fm_no_crop_gt": fm_no_crop_gt,
                "vm_no_crop_gt": vm_no_crop_gt
            }
        elif self.mode=="test":
            meta = {
                "vm_no_crop": vm_no_crop,
                "vm_crop": vm_crop,
                "vm_crop_gt": vm_crop,
                "fm_no_crop": fm_no_crop,
                "fm_crop": fm_crop,
                "img_crop": img_crop,
                "depth_crop": depth_crop,    # 修改： 添加depth  
                "loss_mask": loss_mask,
                "obj_position": obj_position,
                "vm_pad": vm_pad,
                "vm_scale": vm_scale,
                "counts":counts,
                "img_id": img_id,
                "anno_id": anno_id,
                "category_id": category_id,
                # for vq
                # "mask_crop": fm_crop
                "img_no_crop": img,
                "depth_no_crop": depth, # 修改： 添加depth
                "fm_no_crop_gt" : fm_no_crop_gt,
                "vm_no_crop_gt" : vm_no_crop_gt
            }
        return meta
        
    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                res[k] = default_collate(temp_)
            else:
                res[k] = None

        return res

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                collate_fn=self.collate_fn,
                shuffle=True
            )

            for item in sample_loader:
                yield item

    def polys_to_mask(self, polygons, height, width):
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        return mask

    # def data_augmentation(self, mask):
    #     return mask
    
    def data_augmentation(self, mask):
        np.float = float    
        mask = mask.astype(np.float)
        rdv = random.random()
        n_repeat = random.randint(1, 4)
        if rdv <= 0.2:
            mask = cv2.GaussianBlur(mask, (35,35), 11)
        elif rdv > 0.2 and rdv <0.9:
            rdv_1 = random.random()
            rdv_2 = random.random()
            for i in range(n_repeat):
                w = random.randint(5, 13)
                h = random.randint(5, 13)
                kernel = np.ones((w, h), dtype=np.uint8)
                if rdv_1 <= 0.6:
                    mask = cv2.dilate(mask, kernel, 1)
                elif rdv_1 > 0.6 and rdv_1 <= 1.0:
                    mask = cv2.erode(mask, kernel, 1)
                if rdv_2 <= 0.2:
                    mask = cv2.GaussianBlur(mask, (35,35), 11)
        else:
            mask = mask
        return (mask>0.5)
    
    def make_json_dict(self, imgs, anns):
        imgs_dict = {}
        anns_dict = {}
        for ann in anns:
            image_id = ann["image_id"]
            if not image_id in anns_dict:
                anns_dict[image_id] = []
                anns_dict[image_id].append(ann)
            else:
                anns_dict[image_id].append(ann)
        
        for img in imgs:
            image_id = img['id']
            imgs_dict[image_id] = img['file_name']

        return imgs_dict, anns_dict


if __name__ == "__main__":

    train_label = cvb.load("/cpfs/2926428ee2463e44/user/zjy/data/UOAIS-Sim/annotations/coco_anns_uoais_sim_train.json")
    # import pdb; pdb.set_trace()
    anns_dict_list = train_label["annotations"]
    # img_dict_list = train_label["images"]
    count = 0
    for anns_dict in anns_dict_list:
        if anns_dict["iscrowd"] == 1:   count += 1
    print(count)
