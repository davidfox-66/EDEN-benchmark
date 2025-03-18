# [CVPR 2025]EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation

<a href='https://arxiv.org/'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/zhZ524/EDEN/tree/main'><img src='https://img.shields.io/badge/HuggingFace-Model-orange'></a>
<a href='https://bbldCVer.github.io/EDEN/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

This repository is the official implementation of the following paper:

> **EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation**
>
> Zihao Zhang, Haoran Chen, Haoyu Zhao, Guansong Lu, Yanwei Fu, Hang Xu, Zuxuan Wu

<div>
    <h4 align="center">
        <img src="./assets/comparison.jpg">
    </h4>
</div>


## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.jpg"/>
</div><br/>

We introduce EDEN, an enhanced diffusion-based method for high-quality video frame interpolation, addressing the challenging problem of video
frame interpolation with large motion.

Our framework employed a transformer-based tokenizer to compress intermediate frames into compact tokens, enhancing latent representations for the diffusion process. To address multi-scale motion, we incorporated a pyramid feature fusion module and introduced multi-resolution and multi-frame interval fine-tuning to adapt the model to varying motion magnitudes and resolutions. By utilizing a diffusion transformer with temporal attention and a start-end frame difference embedding, EDEN captured complex motion dynamics more effectively. Extensive experiments demonstrated that EDEN achieved state-of-the-art performances on large motion video benchmarks while also reducing computational costs.

## :hammer: Quick Start

### Clone the Repository
```
git clone https://github.com/bbldcver/EDEN.git
cd EDEN
```

### Prepare Environment
```
conda create -n eden python=3.10.13
conda activate eden
pip install -r requirements.txt
```

### Prepare Datasets
Please download the datasets ([LAVIB](https://github.com/alexandrosstergiou/LAVIB?tab=readme-ov-file), [DAVIS](https://drive.google.com/file/d/1tcOoF5DkxJcX7_tGaKgv1B1pQnS7b-xL/view), [DAIN_HD](https://drive.google.com/file/d/1iHaLoR2g1-FLgr9MEv51NH_KQYMYz-FA/view), [SNU_FILM](https://myungsub.github.io/CAIN/)) and store them in the following format.
```
└──── <data directory>/
    ├──── LAVIB/
    |   ├──── annotations/
    |   |   ├──── train.csv/
    |   |   └──── ...
    |   ├──── segments/
    |   |   ├──── 10000_shot0_0_0_0/
    |   |   └──── ...
    |   └──── segments_downsampled/
    |       ├──── 10000_shot0_0_0_0/
    |       └──── ...
    ├──── DAVIS/
    |   ├──── bear/
    |   ├──── bike-packing/
    |   ├──── ...
    |   └──── walking/
    ├──── DAIN_HD/
    |   └──── 544p/
    |       ├──── Sintel_Alley2_1280x544_24_images/
    |       ├──── Sintel_Market5_1280x544_24_images/
    |       ├──── Sintel_Temple_1280x544_24_images/
    |       └──── Sintel_Temple2_1280x544_24_images/
    └──── SNU_FILM/
        ├──── test/
        |   ├──── GOPRO_test/
        |   └──── YouTube_test/
        ├──── test-easy.txt
        ├──── ...
        └──── test-medium.txt
```

### Download Checkpoints
We provide pre-trained model weights, available for download [here](https://huggingface.co/zhZ524/EDEN/tree/main), and recommend saving them in the `checkpoints` folder.

### Inference with EDEN
After downloading the pretrained checkpoints, run the following command to interpolate images or videos with EDEN. The interpolation results are then saved to `interpolation_outputs` folder.
```
CUDA_VISIBLE_DEVICES=0 python inference.py --frame_0_path examples/frame_0.jpg --frame_1_path examples/frame_1.jpg --interpolated_results_dir interpolation_outputs
```

### Evaluation
To evaluate eden, running the following command(change the evaluation dataset in `congfigs/eval_eden.yaml`): 
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch eval.py
```

### Training
EDEN training consists of two stages: **eden_vae** and **eden_dit**. Use the following commands to train each stage:  

- **eden_vae**: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_vae.py`  
- **eden_dit**: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_dit.py`  

Training parameters can be adjusted in `configs/train_vae.yaml` and `configs/train_dit.yaml`. Logs are saved in the `output` folder.

## :fountain_pen: BibTex
``` bibtex
@inproceedings{zhang2025eden,
  title={Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation},
  author={Zhang, Zihao and Chen, Haoran and Zhao, Haoyu and Lu, Guansong and Fu, Yanwei and Xu, Hang and Wu, Zuxuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Acknowledgement
Our code is adapted from [SiT](https://github.com/willisma/SiT) and [LDMVFI](https://github.com/danier97/LDMVFI). Thanks to the team for their impressive work!
