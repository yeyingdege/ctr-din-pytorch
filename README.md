# The Most Complete PyTorch Implementation of "Deep Interest Evolution Network for Click-Through Rate Prediction"

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/yeyingdege/ctr-din-pytorch/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-%237732a8)](https://pytorch.org/get-started/previous-versions/)

This is an unofficial PyTorch implementation of the CTR model DIN with full training and testing pipeline. The model achieved **0.80 AUC** score on Amazon(Books) dataset **without** any parameter / hyperparameter tuning.


## Dataset
You can download the processed Amazon(Books) dataset from [dien](https://github.com/mouna99/dien?tab=readme-ov-file) or [Google Drive](https://drive.google.com/drive/folders/1HK10FUEH_SxBX0oRXioxp0OR0x13G8jz?usp=drive_link). Unzip them and move all files to the "data/" folder.
```
tar -jxvf data.tar.gz
tar -jxvf data1.tar.gz
mv data1/* data/
tar -jxvf data2.tar.gz
mv data2/* data/
```

The "data" folder should have the following files.
- cat_voc.pkl 
- mid_voc.pkl 
- uid_voc.pkl 
- local_train_splitByUser 
- local_test_splitByUser 
- reviews-info
- item-info

## Train model
```
python din/train.py --mode train --ep 5
```

## Test
```
python din/train.py --mode test --model_path path/of/the/model
```

## Acknowledgement
Some code is adapted from [dien](https://github.com/mouna99/dien?tab=readme-ov-file) and [DIN-pytorch](https://github.com/fanoping/DIN-pytorch). Thanks for their great work.
