# 4Diffusion: Multi-view Video Diffusion Model for 4D Generation

| [Project Page]() | [Paper]() |

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


## Quickstart

### Download pre-trained models

Please clone [4DM]() and [ImageDream modelcard](https://huggingface.co/Peng-Wang/ImageDream/resolve/main/sd-v2.1-base-4view-ipmv.pt?download=true) under `./ckpts/`.

### Multi-view Video Generation

To generate multi-view videos, run:
```
bash threestudio/models/imagedream/scripts/demo.sh
```
please configure the `image`(input monocular video path) and `text`(text prompt) in `demo.sh`. The results can be found `threestudio/models/imagedream/4dm`

We use [rembg](https://github.com/danielgatis/rembg) to segment the foreground object for 4D generation.
```
python threestudio/models/imagedream/scripts/remove_bg.py --name yoda
```
`name` denotes the name of folder under `threestudio/models/imagedream/4dm`.

### 4D Generation

To generate 4D content from a monocular video, run:
```
python launch.py --config ./configs/4diffusion.yaml --train \ 
                system.prompt_processor_multi_view.prompt='baby yoda in the style of Mormookiee' \
                system.prompt_processor_multi_view.image_path='./threestudio/models/imagedream/assets/yoda/0_rgba.png' \
                data.multi_view.image_path='./threestudio/models/imagedream/4dm/yoda' \
                system.prompt_processor_multi_view.image_num=8 \
                system.prompt_processor_multi_view.total_num=25 \
                data.multi_view.anchor_view_num=0

```



## Citing

If you find 4D-fy helpful, please consider citing:

```
@article{bah20234dfy,
  author = {Bahmani, Sherwin and Skorokhodov, Ivan and Rong, Victor and Wetzstein, Gordon and Guibas, Leonidas and Wonka, Peter and Tulyakov, Sergey and Park, Jeong Joon and Tagliasacchi, Andrea and Lindell, David B.},
  title = {4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling},
  journal = {arXiv},
  year = {2023},
}
```

## Credits

This code is built on the [threestudio-project](https://github.com/threestudio-project/threestudio) and [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio). Thanks to the maintainers for their contribution to the community!
