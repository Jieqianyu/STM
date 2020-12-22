# Code for "Delving into the Cyclic Mechanism in Semi-supervised Video Object Segmentation"

This repository contains the code for [Delving into the Cyclic Mechanism in Semi-supervised Video Object Segmentation](https://arxiv.org/abs/2010.12176). The result of hybrid training on Youtube-VOS and DAVIS17 can be at most 73.3 J&F score  without gradient correction (improve about 1 J&F score with gradient correction) on DAVIS17 validation set.

## Required Package
- torch
- python-opencv
- pillow
- yaml
- imgaug
- easydict
- progress

## Data Organization

### Youtbe-VOS Organization
To run the training script on youtube-vos dataset, please ensure the data is organized as following format
```
Youtube-VOS
      |----train
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json
      |----valid
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json 
```
Where `JPEGImages` and `Annotations` contain the frames and annotation masks of each video.

### DAVIS Organization

To run the training script on davis16/17 dataset, please ensure the data is organized as following format
```
DAVIS16(DAVIS17)
      |----JPEGImages
      |----Annotations
      |----db_info.yaml
```
Where `JPEGImages` and `Annotations` contain the 480p frames and annotation masks of each video. The `db_info.yaml` contains the meta information of each video sequences and can be found at the davis evaluation [repository](https://github.com/fperazzi/davis-2017/blob/master/data/db_info.yaml).

## Training and Testing
To train the STM network, run following command.
```python
python train.py --gpu ${GPU-IDS}
```
To test the STM network, run following command
```python
python test.py
```
The test results will be saved as indexed png file at `${output}/${valset}`.

Additionally, you can modify some setting parameters in `options.py` to change training configuration.
Reference
The codebase is built based on following works

# Acknowledgement
This codebase borrows the code and structure from [STM-training](https://github.com/lyxok1/STM-Training)