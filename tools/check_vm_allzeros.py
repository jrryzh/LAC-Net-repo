import os
import glob
import cv2
import numpy as np
import imageio

if __name__ == "__main__": 
    dataset_path = "/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth"
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(dataset_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(dataset_path)))
    # can get occluded mask from annotation
    anno_paths = sorted(glob.glob("{}/annotation/*.png".format(dataset_path)))
    amodal_anno_paths = sorted(glob.glob("{}/amodal_annotation/*.png".format(dataset_path)))
    occlusion_anno_paths = sorted(glob.glob("{}/occlusion_annotation/*.png".format(dataset_path)))

    for index in range(len(amodal_anno_paths)):
        anno_file = amodal_anno_paths[index]
        amodal_anno = cv2.imread(anno_file)[...,0]
        img_name = anno_file.split('/')[-1].split('_')[0] + '.png'
        anno_id = int(anno_file.split('/')[-1].split('_')[1].strip('.png'))

        # # 获得img和depth
        # img = cv2.imread(os.path.join("/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/image_color", img_name))
        # height, width, _ = img.shape
        # depth = imageio.imread(os.path.join("/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/disparity", img_name))
        # depth = normalize_depth(depth, min_val=250.0, max_val=1500.0)
        # depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        # depth = inpaint_depth(depth)        # anno_id, img_path = label_info[index].split(",")


        # # ann = anns_dict_list[index]
        # if "learn" in img_name:
        #     img_id = img_name[:-4].strip("learn")
        # else:
        #     img_id = img_name[:-4].strip("test")
        # img_id = int(img_id)
        # # category_id = ann["category_id"]

        # full_mask = amodal_anno>0
        # # fm_no_crop = mask_utils.decode(full_mask)[...,np.newaxis]
        # fm_no_crop = full_mask

        visible_mask = cv2.imread(os.path.join("/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/annotation", img_name))[...,0] == anno_id
        # vm_no_crop = mask_utils.decode(visible_mask)[...,np.newaxis]
        vm_no_crop = visible_mask

        if np.sum(vm_no_crop)==0:
            counts = np.array([0])
            print("DEBUG: all zeors: ",anno_file)