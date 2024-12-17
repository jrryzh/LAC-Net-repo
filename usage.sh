CUDA_VISIBLE_DEVICES=0 python train_single_gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg_linearfusion

CUDA_VISIBLE_DEVICES=0 python test_single_gpu.py --dataset UOAIS --batch 16 --data_type image --path UOAIS_c2f_seg_linearfusion --model rgbd_linearfusion
CUDA_VISIBLE_DEVICES=0 python test_single_gpu.py --dataset OSD --batch 16 --data_type image --path UOAIS_c2f_seg_linearfusion --model rgbd_linearfusion

CUDA_VISIBLE_DEVICES=0 python inference/inference.py --data_type image --path UOAIS_c2f_seg_linearfusion 