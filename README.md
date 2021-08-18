# Robust Object Detection via Instance-Level Temporal Cycle Confusion 
This repo contains the implementation of the ICCV 2021 paper, [Robust Object Detection via Instance-Level Temporal Cycle Confusion](https://arxiv.org/abs/2104.08381).



![Screenshot](figs/cycle_confusion_arch.png)


Building reliable object detectors that are
robust to domain shifts, such as various changes
in context, viewpoint, and object appearances,
is critical for real world applications. In this
work, we study the effectiveness of auxiliary 
self-supervised tasks to improve out-of-distribution generalization of object detectors. Inspired by the principle of maximum entropy, we introduce a novel self-supervised task, instance-level cycle confusion (CycConf), which operates on the region features of the object detectors. For each object, the task is to find the most different object proposals in the adjacent frame in a video and then cycle back to itself for self-supervision. CycConf encourages the object detector to explore invariant structures across instances under various motion, which leads to improved model robustness in unseen domains at test time. We observe consistent out-of-domain performance improvements when training object detectors in tandem with self-supervised tasks on various domain adaptation benchmarks with static images (Cityscapes, Foggy Cityscapes, Sim10K) and large-scale video datasets (BDD100K and Waymo open data). 


## Installation

### Environment
- CUDA 10.2
- Python >= 3.7
- Pytorch >= 1.6
- THe Detectron2 version matches Pytorch and CUDA versions.

### Dependencies

1. Create a virtual env.
- `python3 -m pip install --user virtualenv`
- `python3 -m venv cyc-conf`
- `source cyc-conf/bin/activate`

2. Install dependencies.

- `pip install -r requirements.txt`

- Install Pytorch 1.9

`pip3 install torch torchvision`

Check out the previous Pytorch versions [here](https://pytorch.org/get-started/previous-versions/).

- Install Detectron2
Build Detectron2 from Source (gcc & g++ >= 5.4)
`python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

Or, you can install Pre-built detectron2 (example for CUDA 10.2, Pytorch 1.9)

`python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html`

More details can be found [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Data Preparation

BDD100K
1. Download the BDD100K MOT 2020 dataset (`MOT 2020 Images` and `MOT 2020 Labels`) and the detection labels (`Detection 2020 Labels`) [here](https://bdd-data.berkeley.edu/) and the detailed description is available [here](https://doc.bdd100k.com/download.html). Put the BDD100K data under `datasets/` in this repo. After downloading the data, the folder structure should be like below:
```
├── datasets
│   ├── bdd100k
│   │   ├── images
│   │   │    └── track
│   │   │        ├── train
│   │   │        ├── val
│   │   │        └── test
│   │   └── labels
│   │        ├── box_track_20
│   │        │   ├── train
│   │        │   └── val
│   │        └── det_20
│   │            ├── det_train.json
│   │            └── det_val.json
│   ├── waymo
```

Convert the labels of the MOT 2020 data (train & val sets) into COCO format by running:
```python
python3 datasets/bdd100k2coco.py -i datasets/bdd100k/labels/box_track_20/val/ -o datasets/bdd100k/labels/track/bdd100k_mot_val_coco.json -m track
python3 datasets/bdd100k2coco.py -i datasets/bdd100k/labels/box_track_20/train/ -o datasets/bdd100k/labels/track/bdd100k_mot_train_coco.json -m track
```


2. Split the original videos into different domains (time of day). Run the following command:
```python
python3 -m datasets.domain_splits_bdd100k
```
This script will first extract the domain attributes from the BDD100K detection set and then map them to the tracking set sequences. 
After the processing steps, you would see two additional folders `domain_splits` and `per_seq` under the `datasets/bdd100k/labels/box_track_20`. The domain splits of all attributes in BDD100K detection set can be found at `datasets/bdd100k/labels/domain_splits`.


Waymo
1. Download the Waymo dataset [here](https://waymo.com/open/). Put the Waymo raw data under `datasets/` in this repo. After downloading the data, the folder structure should be like below:
```
├── datasets
│   ├── bdd100k
│   ├── waymo
│   │   └── raw
```

Convert the raw TFRecord data files into COCO format by running:
```python
python3 -m datasets.waymo2coco
```
Note that this script takes a long time to run, be prepared to keep it running for over a day.

2. Convert the BDD100K dataset labels into 3 classes (originally 8). This needs to be done in order to match the 3 classes of the Waymo dataset. Run the following command:
```python
python3 -m datasets.convert_bdd_3cls
```

## Get Started

For joint training,

```python
python3 -m tools.train_net --config-file [config_file] --num-gpus 8
```

For evaluation,

```python
python3 -m tools.train_net --config-file [config_file] --num-gpus [num] --eval-only
```
This command will load the latest checkpoint in the folder. If you want to specify a different checkpoint or evaluate the pretrained checkpoints, you can run 
```python
python3 -m tools.train_net --config-file [config_file] --num-gpus [num] --eval-only MODEL.WEIGHTS [PATH_TO_CHECKPOINT]
```


## Benchmark Results

### Dataset Statistics
| Dataset         | Split | Seq | frames/seq. | boxes | classes |
|-----------------|:-----:|:---:|:-----------:|:-----:|:-------:|
| BDD100K Daytime | train | 757 | 204         | 1.82M | 8       |
|                 | val   | 108 | 204         | 287K  | 8       |
| BDD100K Night   | train | 564 | 204         | 895K  | 8       |
|                 | val   | 71  | 204         | 137K  | 8       |
| Waymo Open Data | train | 798 | 199         | 3.64M | 3       |
|                 | val   | 202 | 199         | 886K  | 3       |


### Out of Domain Evaluation
**BDD100K Daytime to Night.** The base detector is Faster R-CNN with ResNet-50.

| Model               | AP    | AP50  | AP75  | APs  | APm   | APl   | Config | Checkpoint |
|---------------------|-------|-------|-------|------|-------|-------|:------:|:----------:|
| Faster R-CNN        | 17.84 | 31.35 | 17.68 | 4.92 | 16.15 | 35.56 | [link](configs/BDD100K/R50_FPN_daytime.yaml)        |     [link](https://drive.google.com/file/d/1WFeosCd1QPf4EEBSDLpAzpgfyOAvE2EE/view?usp=sharing)       |
| + Rotation          | 18.58 | 32.95 | 18.15 | 5.16 | 16.93 | 36.00 | [link](configs/BDD100K/R50_FPN_daytime_rot.yaml)|    [link](https://drive.google.com/file/d/1q2Wb7e1lHNfmKP1UfpRg1SDYTFa3RJye/view?usp=sharing)         |
| + Jigsaw            | 17.47 | 31.22 | 16.81 | 5.08 | 15.80 | 33.84 | [link](configs/BDD100K/R50_FPN_daytime_jigsaw.yaml)        |   [link](https://drive.google.com/file/d/1MysLkVM5SVEeXdSEFkN3qlNZMsH1ibAi/view?usp=sharing)         |
| + Cycle Consistency | 18.35 | 32.44 | 18.07 | 5.04 | 17.07 | 34.85 | [link](configs/BDD100K/R50_FPN_daytime_cycle_consist.yaml)      |   [link](https://drive.google.com/file/d/1YrX0Ek3M-T1Skzir9ZCy9cnonyhEl8Ju/view?usp=sharing)         |
| + Cycle Confusion   | 19.09 | 33.58 | 19.14 | 5.70 | 17.68 | 35.86 | [link](configs/BDD100K/R50_FPN_daytime_cycle_conf.yaml)      |    [link](https://drive.google.com/file/d/1SiaO9NhuSmZkt5cpf12nKl4HG6JNBOBR/view?usp=sharing)        |


**BDD100K Night to Daytime.**

| Model               | AP    | AP50  | AP75  | APs  | APm   | APl   | Config | Checkpoint |
|---------------------|-------|-------|-------|------|-------|-------|:------:|:----------:|
| Faster R-CNN        | 19.14 | 33.04 | 19.16 | 5.38 | 21.42 | 40.34 | [link](configs/BDD100K/R50_FPN_night.yaml)       |     [link](https://drive.google.com/file/d/1EWexjTuVGok6VjbFzBF8CaotejjIb9cd/view?usp=sharing)       |
| + Rotation          | 19.07 | 33.25 | 18.83 | 5.53 | 21.32 | 40.06 | [link](configs/BDD100K/R50_FPN_night_rot.yaml)    | [link](https://drive.google.com/file/d/1TNWrVyMEhG4iNltCBWK2fgJ0K7P6zTRc/view?usp=sharing) |
| + Jigsaw            | 19.22 | 33.87 | 18.71 | 5.67 | 22.35 | 38.57 | [link](configs/BDD100K/R50_FPN_night_jigsaw.yaml)       |[link](https://drive.google.com/file/d/188qwaDlzmja1_fvqap6wnBAYyEWxifSu/view?usp=sharing)|
| + Cycle Consistency | 18.89 | 33.50 | 18.31 | 5.82 | 21.01 | 39.13 | [link](configs/BDD100K/R50_FPN_night_cycle_consist.yaml)|   [link](https://drive.google.com/file/d/1lAB1wDZ87u6rE13J8cPLuIYf6fEZHjJV/view?usp=sharing)         |
| + Cycle Confusion   | 19.57 | 34.34 | 19.26 | 6.06 | 22.55 | 38.95 | [link](configs/BDD100K/R50_FPN_night_cycle_conf.yaml)       |   [link](https://drive.google.com/file/d/1Byru-K5GYCK57Hr2ItAYvXcmGlNkiJsL/view?usp=sharing)         |

**Waymo Front Left to BDD100K Night.**

| Model               | AP    | AP50  | AP75  | APs  | APm   | APl   | Config | Checkpoint |
|---------------------|-------|-------|-------|------|-------|-------|:------:|:----------:|
| Faster R-CNN        | 10.07 | 19.62 | 9.05 | 2.67 | 10.81 | 18.62 | [link](configs/Waymo/R50_FPN_front_left.yaml) | [link](https://drive.google.com/file/d/1ElCUJwAYMSAqU7Tdxr5B_QiFYLr5Q49b/view?usp=sharing) |
| + Rotation          | 11.34 | 23.12 | 9.65 | 3.53 | 11.73 | 21.60 | [link](configs/Waymo/R50_FPN_front_left_rot.yaml) | [link](https://drive.google.com/file/d/1XersN1sgasNfo7VhZWUddujCcbJhoTDD/view?usp=sharing) |
| + Jigsaw            | 9.86 | 19.93 | 8.40 | 2.77 | 10.53 | 18.82 | [link](configs/Waymo/R50_FPN_front_left_jig.yaml) | [link](https://drive.google.com/file/d/19UQUPz0cpFmOyFxAWj2VX545qk_0Qc9J/view?usp=sharing) |
| + Cycle Consistency | 11.55 | 23.44 | 10.00 | 2.96 | 12.19 | 21.99 | [link](configs/Waymo/R50_FPN_front_left_cycle_cons.yaml) | [link](https://drive.google.com/file/d/1Tqzbxx0B7DnjNs6gXls03aXNsy_vmN7G/view?usp=sharing) |
| + Cycle Confusion   | 12.27 | 26.01 | 10.24 | 3.44 | 12.22 | 23.56 | [link](configs/Waymo/R50_FPN_front_left_cycle_conf.yaml) | [link](https://drive.google.com/file/d/1dKOgcLUpas6MXbir41YCwO-aISgTazpG/view?usp=sharing) |

**Waymo Front Right to BDD100K Night.**

| Model               | AP   | AP50  | AP75 | APs  | APm   | APl   | Config | Checkpoint |
|---------------------|------|-------|------|------|-------|-------|:------:|:----------:|
| Faster R-CNN        | 8.65 | 17.26 | 7.49 | 1.76 | 8.29 | 19.99 | [link](configs/Waymo/R50_FPN_front_right.yaml)  | [link](https://drive.google.com/file/d/1uScVSpFe4zLuGfyDiAApBO7nqhP-EvHK/view?usp=sharing) |
| + Rotation          | 9.25 | 18.48 | 8.08 | 1.85 | 8.71 | 21.08 | [link](configs/Waymo/R50_FPN_front_right_rot.yaml)  | [link](https://drive.google.com/file/d/14nOrpORe6ohNm162g9ThspziTUnbw4Dn/view?usp=sharing) |
| + Jigsaw            | 8.34 | 16.58 | 7.26 | 1.61 | 8.01 | 18.09 | [link](configs/Waymo/R50_FPN_front_right_jig.yaml)  | [link](https://drive.google.com/file/d/1V30OLYoAKOhGg6RH5BtXP5Xdz2YQVXGE/view?usp=sharing) |
| + Cycle Consistency | 9.11 | 17.92 | 7.98 | 1.78 | 9.36 | 19.18 | [link](configs/Waymo/R50_FPN_front_right_cycle_cons.yaml)  | [link](https://drive.google.com/file/d/1CfpnRwakAiC6OmcOUHacyxlkd1gBUYZC/view?usp=sharing) |
| + Cycle Confusion   | 9.99 | 20.58 | 8.30 | 2.18 | 10.25 | 20.54 | [link](configs/Waymo/R50_FPN_front_right_cycle_conf.yaml)  | [link](https://drive.google.com/file/d/13mCXi-lEU-AwJsMPACH5PjQT34Em2PPZ/view?usp=sharing) |



## Citation
If you find this repository useful for your publications, please consider citing our paper.

```
@article{wang2021robust,
  title={Robust Object Detection via Instance-Level Temporal Cycle Confusion},
  author={Wang, Xin and Huang, Thomas E and Liu, Benlin and Yu, Fisher and Wang, Xiaolong and Gonzalez, Joseph E and Darrell, Trevor},
  journal={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
