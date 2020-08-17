# Generation of Virtual Dual Energy Images from Standard Single-Shot Radiographs using Multi-scale and Conditional Adversarial Network

Bo Zhou, Xunyu Lin, Brendan Eck, Jun Hou, David L. Wilson

Asian Conference on Computer Vision (ACCV), 2018

[[Paper](https://arxiv.org/pdf/1810.09354.pdf)]

This repository contains the PyTorch implementation of MCA-Net for DE bone image generation. \
Pre-trained model is available upon request via: \
https://drive.google.com/file/d/1agAASv1B5Uecxh9uyt4q2i-6NTDRLPYc/view?usp=sharing

The code and model are for research use only. 

We provide an example case in the './example_data/'

### Citation
If you use this code for your research or project, please cite:

    @inproceedings{zhou2018generation,
      title={Generation of virtual dual energy images from standard single-shot radiographs using multi-scale and conditional adversarial network},
      author={Zhou, Bo and Lin, Xunyu and Eck, Brendan and Hou, Jun and Wilson, David},
      booktitle={Asian Conference on Computer Vision},
      pages={298--313},
      year={2018},
      organization={Springer}
    }


### Environment and Dependencies
Requirements:
* Python 3.7
* Pytorch 0.4.1
* scipy
* scikit-image
* opencv-python
* tqdm

Our code has been tested with Python 3.7, Pytorch 0.4.1, CUDA 10.0 on Ubuntu 18.04.


### Dataset Setup
    .
    example_data
    ├── Train                   # contain training files (IB:bone image / IS:soft-tissue image / IH:standard chest x-ray image)
    │   ├── IB
    │   │   ├── IB_1.png         
    │   │   ├── IB_2.png 
    │   │   ├── ...         
    │   │   └── IB_N.png 
    │   │   
    │   ├── IS
    │   │   ├── IS_1.png         
    │   │   ├── IS_2.png 
    │   │   ├── ...         
    │   │   └── IS_N.png 
    │   │   
    │   ├── IH
    │   │   ├── IH_1.png         
    │   │   ├── IH_2.png 
    │   │   ├── ...         
    │   │   └── IH_N.png 
    │   └── ...
    │
    │
    ├── Test                    # contain test files (IB:bone image / IS:soft-tissue image / IH:standard chest x-ray image)
    │   ├── IB
    │   │   ├── IB_1.png         
    │   │   ├── IB_2.png 
    │   │   ├── ...         
    │   │   └── IB_N.png 
    │   │   
    │   ├── IS
    │   │   ├── IS_1.png         
    │   │   ├── IS_2.png 
    │   │   ├── ...         
    │   │   └── IS_N.png 
    │   │   
    │   ├── IH
    │   │   ├── IH_1.png         
    │   │   ├── IH_2.png 
    │   │   ├── ...         
    │   │   └── IH_N.png 
    │   └── ...
    │            
    └── ...

Each .png is an image data and intensity normalized to between 0~1. IB_N.png / IS_N.png / IH_N.png should contain paired imaging data for the same patient.

### To Run Our Code
- Train the model
```bash
python train.py --experiment_name 'train_bone_msunet' --model_type 'model_bone' --dataset 'DE' --data_root './example_data/' --net_G 'msunet' --net_D 'patchGAN' --wr_recon 50 --batch_size 2 --lr 1e-4 --AUG
```
where \
`--experiment_name` provides the experiment name for the current run, and save all the corresponding results under the experiment_name's folder. \
`--data_root`  provides the data folder directory (with structure illustrated above). \
`--AUG` adds for using data augmentation option (rotation, random cropping, scaling). \
Other hyperparameters can be adjusted in the code as well.

- Test the model
```bash
python test.py --resume './output/train_bone_msunet/checkpoints/model_best.pt' --experiment_name 'test_bone_msunet' --model_type 'model_bone' --dataset 'DE' --data_root './example_data/' --net_G 'msunet' --net_D 'patchGAN'
```
where \
`--resume` defines which checkpoint for testing and evaluation. The 'model_best.pt' is available upon request.  \
The test will output an eval.mat containing model's input and prediction for evaluation in the '--experiment_name' folder.

Sample training/test scripts are provided under './scripts/' and can be directly executed.

### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```