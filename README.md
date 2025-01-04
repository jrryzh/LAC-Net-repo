# LAC-Net: Linear-Fusion Attention-Guided Convolutional Network for Accurate Robotic Grasping Under the Occlusion

This repository contains the source code for my paper "LAC-Net: Linear-Fusion Attention-Guided Convolutional Network for Accurate Robotic Grasping Under the Occlusion," accepted at IROS2024.

## TL;DR

This is a state-of-the-art method that uses RGB-D data to predict amodal masks, enabling accurate grasping of occluded objects in complex environments. It is particularly effective in scenarios where objects are hidden beneath fine foam or buried in sand, such as on a beach.

## Introduction

LAC-Net is a novel convolutional neural network designed to address the challenge of perceiving complete object shapes through visual perception for accurate robotic grasping in occluded environments. While prior studies have focused on segmenting visible parts of objects, LAC-Net explores amodal segmentation to infer occluded parts, enhancing robotic grasping abilities in cluttered scenes. By leveraging a linear-fusion strategy to effectively combine semantic features from RGB images and geometric information from depth images, LAC-Net uses the prior visible mask as an attention map to guide the network in recovering complete object masks. This approach allows for the selection of more accurate and robust grasp points, achieving state-of-the-art performance across various datasets and demonstrating feasibility and robustness in real-world robot experiments.

## Repository Structure

- `src/`: Contains the main model code and related modules.
  - `image_model_depth_linearfusion.py`: Implements the core model of LAC-Net.
  - `image_component.py`: Includes image encoders and fusion modules.
- `data/`: Data loaders and dataset-related code.
  - `dataloader_transformer.py`: Implementation of the data loader.
- `utils/`: Utility functions and helper modules.
  - `pytorch_optimization.py`: Optimizers and learning rate schedulers.
  - `loss.py`: Implementation of loss functions.
  - `utils.py`: General utility functions.
- `test_single_gpu.py`: Script for testing on a single GPU.
- `.gitignore`: Git ignore file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LAC-Net.git
   cd LAC-Net
   ```

2. Create the environment:
   ```bash
   conda env create -f environment.yml
   ```

## Usage

### Training the Model

1. Configure training parameters:
   Locate the appropriate dataset configuration file in the `configs` folder (e.g., `LAC_Net_UOAIS.yml`) and modify parameters as needed.

2. Run the training script:
   ```bash
   python train_single_gpu.py --path <your_experiment_path> --dataset <your_dataset>
   ```

### Testing the Model

1. Configure testing parameters:
   Locate the appropriate dataset configuration file in the `configs` folder (e.g., `LAC_Net_UOAIS.yml`) and modify parameters as needed.

2. Run the testing script:
   ```bash
   python test_single_gpu.py --path <your_experiment_path> --dataset <your_dataset>
   ```

## Citation

If you use LAC-Net in your research, please cite our paper:
```
@misc{zhang2024lacnetlinearfusionattentionguidedconvolutional,
      title={LAC-Net: Linear-Fusion Attention-Guided Convolutional Network for Accurate Robotic Grasping Under the Occlusion}, 
      author={Jinyu Zhang and Yongchong Gu and Jianxiong Gao and Haitao Lin and Qiang Sun and Xinwei Sun and Xiangyang Xue and Yanwei Fu},
      year={2024},
      eprint={2408.03238},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2408.03238}, 
}
```
