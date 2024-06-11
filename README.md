# 4Diffusion: Multi-view Video Diffusion Model for 4D Generation

| [Project Page](https://aejion.github.io/4diffusion) | [Paper](http://arxiv.org/abs/2405.20674) |

Official code for 4Diffusion: Multi-view Video Diffusion Model for 4D Generation.

The paper presents a novel 4D generation pipeline, namely 4Diffusion, aimed at generating spatial-temporally consistent 4D content from a monocular video. We design a multi-view video diffusion model 4DM to capture multi-view spatial-temporal correlations for multi-view video generation.

## Installation Requirements

The code is compatible with python 3.10.0 and pytorch 2.0.1. To create an anaconda environment named `4diffusion` with the required dependencies, run:

```
conda create -n 4diffusion python==3.10.0
conda activate 4diffusion

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 4D Data

We filter out animated 3D shapes from the vast 3D data corpus of [Objaverse-1.0](https://objaverse.allenai.org/objaverse-1.0/). We provide ids of the curated data in `dataset/uid.npy`. We will also release the rendered multi-view videos (To be uploaded) for future works.


## Quickstart

### Download pre-trained models

Please download [4DM](https://drive.google.com/drive/folders/19k3p2CfzQ6ArqpDNOy73RJeJhWfNs4i6?usp=sharing) and [ImageDream modelcard](https://huggingface.co/Peng-Wang/ImageDream/resolve/main/sd-v2.1-base-4view-ipmv.pt?download=true) and put them under `./ckpts/`.

### Multi-view Video Generation

To generate multi-view videos, run:
```
bash threestudio/models/imagedream/scripts/demo.sh
```
please configure the `image`(input monocular video path), `text`(text prompt), and `num_video_frames`(number of frames of input monocular video) in `demo.sh`. The results can be found in `threestudio/models/imagedream/4dm`.

We use [rembg](https://github.com/danielgatis/rembg) to segment the foreground object for 4D generation.
```
# name denotes the folder's name under threestudio/models/imagedream/4dm
python threestudio/models/imagedream/scripts/remove_bg.py --name yoda
```


### 4D Generation

To generate 4D content from a monocular video, run:
```
# system.prompt_processor_multi_view.prompt: text prompt
# system.prompt_processor_multi_view.image_path: monocular video path
# data.multi_view.image_path: anchor video path (anchor loss in Sec3.3)
# system.prompt_processor_multi_view.image_num: number of frames for training, default: 8
# system.prompt_processor_multi_view.total_num: number of frames of input monocular video
# data.multi_view.anchor_view_num: anchor view for anchor loss. 0: 0 azimuth; 1: 90 azimuth; 2: 180 azimuth; 3: 270 azimuth
python launch.py --config ./configs/4diffusion.yaml --train \ 
                system.prompt_processor_multi_view.prompt='baby yoda in the style of Mormookiee' \
                system.prompt_processor_multi_view.image_path='./threestudio/models/imagedream/assets/yoda/0_rgba.png' \
                data.multi_view.image_path='./threestudio/models/imagedream/4dm/yoda' \
                system.prompt_processor_multi_view.image_num=8 \
                system.prompt_processor_multi_view.total_num=25 \
                data.multi_view.anchor_view_num=0
```
The results can be found in `outputs/4diffusion`.


## Citing

If you find 4Diffusion helpful, please consider citing:

```
@article{zhang20244diffusion,
  title={4Diffusion: Multi-view Video Diffusion Model for 4D Generation},
  author={Zhang, Haiyu and Chen, Xinyuan and Wang, Yaohui and Liu, Xihui and Wang, Yunhong and Qiao, Yu},
  journal={arXiv preprint arXiv:2405.20674},
  year={2024}
}
```

## Credits

This code is built on the [threestudio-project](https://github.com/threestudio-project/threestudio), [4D-fy](https://github.com/sherwinbahmani/4dfy), and [ImageDream](https://github.com/bytedance/ImageDream). Thanks to the maintainers for their contribution to the community!
