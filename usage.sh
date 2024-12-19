CUDA_VISIBLE_DEVICES=0 python train_single_gpu.py --dataset UOAIS --batch 32 --path UOAIS_LAC_Net

CUDA_VISIBLE_DEVICES=0 python test_single_gpu.py --dataset UOAIS --batch 16 --path UOAIS_LAC_Net

CUDA_VISIBLE_DEVICES=0 python inference/inference.py --path UOAIS_LAC_Net