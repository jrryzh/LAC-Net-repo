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

def inf_dataloader(img_path, depth_path, visible_mask_path, enlarge_coef=2, patch_h=256, patch_w=256, dtype=torch.float32, device="cpu"):
    # 所有需要的输入
    # image, depth, visible_mask, amodal_mask(假的或不给)，bbox
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width, _ = img.shape

    depth = imageio.imread(depth_path).astype(np.float32)
    depth_min, depth_max = [depth.min(), depth.max()]
    depth[depth > depth_max] = depth_max
    depth[depth < depth_min] = depth_min
    depth = (depth - depth_min) / (depth_max - depth_min) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))

    anno_id = 0
    img_id = 0
    category_id = 1

    # visible_mask = ann["visible_mask"]
    # vm_no_crop = mask_utils.decode(visible_mask)[...,np.newaxis]
    # 在这里想办法读入visible mask
    # todo
    vm_no_crop = imageio.imread(visible_mask_path)
    vm_no_crop = vm_no_crop > 0

    rows, cols = np.where(vm_no_crop)
    x_min, x_max = np.min(rows), np.max(rows)
    y_min, y_max = np.min(cols), np.max(cols)
    # y_max, x_max = y_min + w, x_min + h
    # y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max) 
    

    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_len = int((x_max - x_min) * enlarge_coef)
    y_len = int((y_max - y_min) * enlarge_coef)
    x_min = max(0, x_center - x_len // 2)
    x_max = min(height, x_center + x_len // 2)
    y_min = max(0, y_center - y_len // 2)
    y_max = min(width, y_center + y_len // 2)
    
    vm_crop = vm_no_crop[x_min:x_max+1, y_min:y_max+1].astype(bool)
    img_crop = img[x_min:x_max+1, y_min:y_max+1]
    depth_crop = depth[x_min:x_max+1, y_min:y_max+1]

    h, w = vm_crop.shape[:2]
    m = transform.rescale(vm_crop, (patch_h/h, patch_w/w))
    cur_h, cur_w = m.shape[:2]
    to_pad = ((0, max(patch_h-cur_h, 0)), (0, max(patch_w-cur_w, 0)))
    m = np.pad(m, to_pad)[:patch_h, :patch_w]
    vm_crop = m[np.newaxis, ...]

    img_ = transform.rescale(img_crop, (patch_h/h, patch_w/w, 1))
    cur_h, cur_w = img_.shape[:2]
    to_pad = ((0, max(patch_h-cur_h, 0)), (0, max(patch_w-cur_w, 0)), (0, 0))
    img_ = np.pad(img_, to_pad)[:patch_h, :patch_w, :3]
    img_crop = img_

    # 修改：添加depth
    depth_ = transform.rescale(depth_crop, (patch_h/h, patch_w/w, 1))
    cur_h, cur_w = depth_.shape[:2]
    to_pad = ((0, max(patch_h-cur_h, 0)), (0, max(patch_w-cur_w, 0)), (0, 0))
    depth_ = np.pad(depth_, to_pad)[:patch_h, :patch_w, :3]
    depth_crop = depth_

    obj_position = np.array([x_min, x_max, y_min, y_max])
    vm_pad = np.array([max(patch_h-cur_h, 0), max(patch_w-cur_w, 0)])
    vm_scale = np.array([patch_h/h, patch_w/w])

    vm_no_crop = vm_no_crop[np.newaxis, ...]

    # data augmentation
    vm_crop_aug = data_augmentation(vm_crop[0])[np.newaxis, ...]

    obj_position = torch.from_numpy(obj_position).to(dtype).to(device)
    vm_pad = torch.from_numpy(vm_pad).to(dtype).to(device)
    vm_scale = torch.from_numpy(vm_scale).to(dtype).to(device)

    vm_crop = torch.from_numpy(vm_crop).to(dtype).to(device)
    vm_crop_aug = torch.from_numpy(vm_crop_aug).to(dtype).to(device)
    img_crop = torch.from_numpy(img_crop).to(dtype).to(device)
    img = torch.from_numpy(img).to(dtype).to(device)
    # 修改： 添加depth
    depth_crop = torch.from_numpy(depth_crop).to(dtype).to(device)
    depth = torch.from_numpy(depth).to(dtype).to(device)
    vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(dtype).to(device)
    
    img_id = torch.from_numpy(np.array(img_id)).to(dtype).to(device)
    anno_id = torch.from_numpy(np.array(anno_id)).to(dtype).to(device)
    category_id = torch.from_numpy(np.array(category_id)).to(dtype).to(device)

    meta = {
        "vm_crop": vm_crop.unsqueeze(0), # torch.Size([1, 256, 256])
        "vm_crop_gt": vm_crop.unsqueeze(0), # torch.Size([1, 256, 256])
        "img_crop": img_crop.unsqueeze(0), # torch.Size([256, 256, 3])
        "depth_crop": depth_crop.unsqueeze(0),    # torch.Size([256, 256, 3])
        "obj_position": obj_position.unsqueeze(0), # torch.Size([4])
        "vm_pad": vm_pad.unsqueeze(0), # torch.Size([2])
        "vm_no_crop": vm_no_crop.unsqueeze(-1).unsqueeze(0) # torch.Size([1, 480, 640, 1])
        # for vq
        # "mask_crop": fm_crop
        # "img_no_crop": img,
        # "depth_no_crop": depth, # 修改： 添加depth
        # "vm_no_crop_gt" : vm_no_crop_gt
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


def polys_to_mask(polygons, height, width):
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask


def data_augmentation(mask):
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

def make_json_dict(imgs, anns):
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