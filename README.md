## Module for predicting semantic edges between point clouds based on VL-SAT method for 3DSSG.


# Introduction
This is repository based on the source code of paper **_VL-SAT: Visual-Linguistic Semantics Assisted Training for 3D Semantic Scene Graph Prediction in Point Cloud_** (CVPR 2023 Highlight).

[[arxiv]](https://arxiv.org/pdf/2303.14408.pdf)  [[code]](https://github.com/wz7in/CVPR2023-VLSAT)  [[checkpoint]](https://drive.google.com/file/d/1_C-LXRlSobupApb-JsajKG5oxKnfKgdx/view?usp=sharing)

# Installation
```bash
conda create -n vlsat python=3.8
conda activate vlsat
pip install -r requirement.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-geometric
pip install git+https://github.com/openai/CLIP.git
```
# Prepare the data

0. Download the pretrained checkpoint: [[checkpoint]](https://drive.google.com/file/d/1_C-LXRlSobupApb-JsajKG5oxKnfKgdx/view?usp=sharing), unzip folder.

1. Create a folder with saved point clouds.

2. Change paths in the main function [vlsat_inference.py](vlsat_inference.py) to your environment. Change the way of reading point clouds, if necessary.

# Run Code
```bash
# Base inference with wrapper class EdgePredictor.
python vlsat_inference.py
```

You can visualize point clouds with [visualize_pointclouds.ipynb](visualize_pointclouds.ipynb) notebook.