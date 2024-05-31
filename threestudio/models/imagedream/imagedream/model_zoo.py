""" Utiliy functions to load pre-trained models more easily """
import os
import pkg_resources
from omegaconf import OmegaConf

import torch
from huggingface_hub import hf_hub_download

from .ldm.util import instantiate_from_config


PRETRAINED_MODELS = {
    "sd-v2.1-base-4view-ipmv": {
        "config": "sd_v2_base_ipmv.yaml",
        "repo_id": "Peng-Wang/ImageDream",
        "filename": "sd-v2.1-base-4view-ipmv.pt",
    },
    "sd-v2.1-base-4view-ipmv-local": {
        "config": "sd_v2_base_ipmv_local.yaml",
        "repo_id": "Peng-Wang/ImageDream",
        "filename": "sd-v2.1-base-4view-ipmv-local.pt",
    },
}


def get_config_file(config_path):
    cfg_file = pkg_resources.resource_filename(
        "imagedream", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError(f"Config {config_path} not available!")
    return cfg_file


def build_model(model_name, config_path=None, ckpt_path=None, cache_dir=None):
    if (config_path is not None) and (ckpt_path is not None):
        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        new_ckpt = ckpt
        if config.model.params.unet_config.params.use_motion_module:
            new_ckpt = {}
            keys = list(ckpt.keys())
            for key in keys:
                if 'middle_block.2' in key:
                    new_ckpt[key.replace('middle_block.2', 'middle_block.3')] = ckpt[key]
                elif 'output_blocks.2.1' in key:
                    new_ckpt[key.replace('output_blocks.2.1', 'output_blocks.2.2')] = ckpt[key]
                elif 'output_blocks.5.2' in key:
                    new_ckpt[key.replace('output_blocks.5.2', 'output_blocks.5.3')] = ckpt[key]
                elif 'output_blocks.8.2' in key:
                    new_ckpt[key.replace('output_blocks.8.2', 'output_blocks.8.3')] = ckpt[key]
                else:
                    new_ckpt[key] = ckpt[key]

        missing, unexpected = model.load_state_dict(new_ckpt, strict=False)
        print(f"### missing keys: {len(missing)}; \n### unexpected keys: {len(unexpected)};")
        ckpt = torch.load(
            './ckpts/check.ckpt',
            map_location="cpu")['state_dict']
        new_ckpt = {}
        keys = list(ckpt.keys())
        for key in keys:
            new_ckpt[key.replace('module.', '')] = ckpt[key]
        missing, unexpected = model.model.diffusion_model.load_state_dict(new_ckpt, strict=False)
        print(f"### missing keys: {len(missing)}; \n### unexpected keys: {len(unexpected)};")
        return model

    if not model_name in PRETRAINED_MODELS:
        raise RuntimeError(
            f"Model name {model_name} is not a pre-trained model. Available models are:\n- "
            + "\n- ".join(PRETRAINED_MODELS.keys())
        )
    model_info = PRETRAINED_MODELS[model_name]

    # Instiantiate the model
    print(f"Loading model from config: {model_info['config']}")
    config_file = get_config_file(model_info["config"])
    config = OmegaConf.load(config_file)
    model = instantiate_from_config(config.model)

    # Load pre-trained checkpoint from huggingface
    if not ckpt_path:
        ckpt_path = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["filename"],
            cache_dir=cache_dir,
        )
        print(f"Loading model from cache file: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    return model
