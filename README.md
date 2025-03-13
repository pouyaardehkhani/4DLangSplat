# [CVPR2025] 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models
[Wanhua Li*](https://li-wanhua.github.io/), [Renping Zhou*](https://github.com/zrporz), [Jiawei Zhou](https://joezhouai.com/), [Yingwei Song](https://github.com/wrencanfly), [Johannes Herter](https://www.linkedin.com/in/johannes-herter-48a549155/), [Minghan Qin](https://github.com/minghanqin), [Gao Huang†](https://www.gaohuang.net/), [Hanspeter Pfister†](https://seas.harvard.edu/person/hanspeter-pfister) \
(* indicates equal contribution, † means Co-corresponding author) \
| [Project page](https://4d-langsplat.github.io) | [Full Paper]() | [Video](https://youtu.be/L2OzQ91eRG4) |\
| Datasets Annotations | [Google Drive](https://drive.google.com/drive/folders/1C-ciHn38vVd47TMkx2-93EUpI0z4ZdZW?usp=sharing) | [BaiduWangpan](https://pan.baidu.com/s/1ZMOk0UFQ39WJ7TtTXy9gkA?pwd=g9rg)\
| Pretrained Model | [Google Drive](https://drive.google.com/drive/folders/1-G8I5cJCD66fjpvejUzF9QPRJU_GNxj0?usp=sharing) | [BaiduWangpan](https://pan.baidu.com/s/1TmBW1ZjZfjLQTGxpDXZzlg?pwd=3kmw)\
| Pregenerated Point Clouds by COLMAP | [Google Drive](https://drive.google.com/drive/folders/1_JOObfpXrCq3v_NYKwDt6vRHIbb0oVek?usp=sharing) | [BaiduWangpan](https://pan.baidu.com/s/15jDvS-zSW7pfdvzdwP32mQ?pwd=9y2u)
<img src="./assets/teaser.png"> 
This repository contains the official implementation of the paper "4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models" (CVPR 2025).
## BibTeX

## Cloning the Repository
The repository contains submodules, thus please check it out with
```bash
git clone git@github.com:zrporz/4DLangSplat.git --recursive
```

## Setup
4D LangSplat uses the following software versions:
- Python 3.10
- CUDA 12.4
- GCC 10.2.0

On default, run the following commands to install the relative packages
```bash
conda create -n 4DLangSplat python=3.10
conda activate 4DLangSplat
pip install -r requirements.txt
### submodules for gaussian rasterization ###
pip install -e submodules/simple-knn
pip install -e submodules/4d-langsplat-rasterization
### submodules for generate segmentation map ###
pip install -e submodules/4d-langsplat-tracking-anything-with-deva
```

## Prepare Datasets
Our models are trained and evaluated on [HyperNeRF](https://github.com/google/hypernerf) and [Neu3D](https://github.com/facebookresearch/Neural_3D_Video) datasets. Please follow their instructions to prepare your dataset, or run the following commands:
```bash
bash scripts/download_hypernerf.sh data/hypernerf
bash scripts/download_neu3d.sh data/neu3d
```

To evaluate the rendering results, we use [RoboFlow](https://roboflow.com/) to annotate the datasets. The annotations can be accessed through this link: [Download the Annotations](https://drive.google.com/drive/folders/1C-ciHn38vVd47TMkx2-93EUpI0z4ZdZW?usp=sharing). \
Follow [4DGaussians](https://github.com/hustvl/4DGaussians), we use COLMAP to generate the point clouds. Please follow their pipeline, or use ours: [Download the Point Clouds](https://drive.google.com/drive/folders/1_JOObfpXrCq3v_NYKwDt6vRHIbb0oVek?usp=sharing)

Then put them under `data/<hypernerf or neu3d>/<dataset name>`. You need to ensure that the data folder is organized as follows:
```
|——data
|   | hypernerf
|       | americano
|           |——annotations
|               |——train
|               |——README
|               |——video_annotations.json
|           |——camera
|           |——rgb
|               |——1x
|                   |——000001.png
|                   ...
|               |——2x        
|               ...
|           |——dataset.json
|           |——metadata.json
|           |——points.npy
|           |——scene.json
|           |——points3D_downsample2.ply
|       |——chickchicken
|       ...
|   | neu3d
|       | coffee_martini
|           |——annotations
|               |——train
|               |——README
|           |——cam00
|               |——images
|                   |——0000.png
|                   ...
|           |——cam01
|           ...
|           |——cam00.mp4
|           |——cam01.mp4
|           ...
|           |——poses_bounds.npy
|           |——points3D_downsample2.ply
|      |——cur_roasted_beef
|      ...
```

## QuickStart
We provide the pretrained checkpoints of gaussian model and autoencoder: [Download Pretrained Checkpoint](https://drive.google.com/drive/folders/1-G8I5cJCD66fjpvejUzF9QPRJU_GNxj0?usp=sharing).

For HyperNeRF dataset, take `americano` as an example. Put checkpoint folder upder the  `output/hypernerf/americano` and run the following commands for rendering and evaluation
```bash
bash scripts/render-hypernerf.sh
bash scripts/eval-hypernerf.sh
```
For Neu3D dataset, take `coffee_martini` as an example. Put checkpoint folder under the  `output/neu3d/coffee_martini` and run the following commands for rendering and evaluation
```bash
bash scripts/render-neu3d.sh
bash scripts/eval-neu3d.sh
```

The evaluation results will be saved under `eval/eval_results`.



## TODO list
- [x] release the code of the 4d-langsplat-rasterization
- [x] release the code of the 4d-langsplat-tracking-anything-with-deva
- [x] release the code of the evaluation
- [x] release the code of the autoencoder
- [ ] release the code of preprocessing
- [ ] release the code of training
- [x] release the the pretrained model
- [ ] release the preprocessed dataset
- [ ] update the arxiv link
