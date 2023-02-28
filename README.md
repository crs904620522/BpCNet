# BpCNet

###### *PyTorch implementation of AAAI 2023 paper: "Take Your Model Further: A General Post-refinement Network for Light Field Disparity Estimation via BadPix Correction"*.

[paper waiting]()

#### Requirements

- python 3.6
- pytorch 1.8.0
- ubuntu 18.04

### Installation

First you have to make sure that you have all dependencies in place. 

You can create an anaconda environment called BpCNet using

```
conda env create -f BpCNet.yaml
conda activate BpCNet
```

### Demo

Here, we train BpCNet on [OACC-Net](https://github.com/YingqianWang/OACC-Net) and perform refinement as a demo.

##### Dataset: 

Light Field Dataset: We use [HCI 4D Light Field Dataset](https://lightfield-analysis.uni-konstanz.de/) for training and test. Please first download light field datasets, and put them into corresponding folders in ***data/HCInew***.

Initial disparity map: We perform BpCNet on other LF disparity methods for refinement, you can provide initial data and put them into ***data/CoarseData*** or use the demo we provided.

##### To train, run:

```
python train.py --config configs/HCInew/BpCNet.yaml 
```

##### To generate, run:

```
python generate.py --config configs/pretrained/HCInew/BpCNet_pretrained.yaml 
```



**If you find our code or paper useful, please consider citing:**

**waiting**





