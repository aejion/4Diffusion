import os
import sys
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch

from threestudio.models.imagedream.imagedream.camera_utils import get_camera
from threestudio.models.imagedream.imagedream.ldm.util import (
    instantiate_from_config,
    set_seed,
    add_random_background
)
from threestudio.models.imagedream.imagedream.ldm.models.diffusion.ddim import DDIMSampler
from threestudio.models.imagedream.imagedream.model_zoo import build_model
from torchvision import transforms as T

import torchvision
from einops import rearrange
import imageio

def i2i(
    model,
    image_size,
    prompt,
    uc,
    sampler,
    ip=None,
    step=20,
    scale=5.0,
    batch_size=8,
    ddim_eta=0.0,
    dtype=torch.float32,
    device="cuda",
    camera=None,
    num_frames=4,
    pixel_control=False,
    transform=None
):
    """ The function supports additional image prompt.
    Args:
        model (_type_): the image dream model
        image_size (_type_): size of diffusion output
        prompt (_type_): text prompt for the image
        uc (_type_): _description_
        sampler (_type_): _description_
        ip (Image, optional): the image prompt. Defaults to None.
        step (int, optional): _description_. Defaults to 20.
        scale (float, optional): _description_. Defaults to 7.5.
        batch_size (int, optional): _description_. Defaults to 8.
        ddim_eta (float, optional): _description_. Defaults to 0.0.
        dtype (_type_, optional): _description_. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to "cuda".
        camera (_type_, optional): _description_. Defaults to None.
        num_frames (int, optional): _description_. Defaults to 4
        pixel_control: whether to use pixel conditioning. Defaults to False.
    """
    if type(prompt) != list:
        prompt = [prompt]

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}

        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        if ip is not None:
            ip_embed = model.get_learned_image_conditioning(ip).to(device)
            ip_ = ip_embed.repeat(batch_size // 8, 1, 1)
            c_["ip"] = ip_
            uc_["ip"] = torch.zeros_like(ip_)

        if pixel_control:
            assert camera is not None
            # ip = transform(ip).to(device)
            ip = ((ip / 255. - 0.5) * 2).to(device).permute(0, 3, 1, 2)
            ip_img = model.get_first_stage_encoding(
                model.encode_first_stage(ip)
            )
            c_["ip_img"] = ip_img
            uc_["ip_img"] = torch.zeros_like(ip_img)

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None,
        )
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


class ImageDreamDiffusion():
    def __init__(self, args) -> None:
        assert args.mode in ["pixel", "local"]
        assert args.num_frames % 4 == 1 if args.mode == "pixel" else True

        set_seed(args.seed)
        dtype = torch.float16 if args.fp16 else torch.float32
        device = args.device
        batch_size = max(4, args.num_frames) * 8

        print("load image dream diffusion model ... ")
        model = build_model(args.model_name,
                            config_path=args.config_path,
                            ckpt_path=args.ckpt_path)
        # quit()
        model.device = device
        model.to(device)
        model.eval()

        neg_texts = "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear."
        # neg_texts = ""
        sampler = DDIMSampler(model)
        uc = model.get_learned_conditioning([neg_texts]).to(device)
        print("image dream model load done . ")

        # pre-compute camera matrices
        if args.use_camera:
            camera = get_camera(
                num_frames=4,
                elevation=0,
                azimuth_start=90,
                azimuth_span=360,
                extra_view=(args.mode == "pixel")
            )
            camera = camera.repeat(batch_size // args.num_frames, 1).to(device)
        else:
            camera = None

        self.image_transform = T.Compose(
            [
                T.Resize((args.size, args.size)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.args = args
        self.model = model
        self.sampler = sampler
        self.uc = uc
        self.camera = camera

    def diffuse(self, t, ip, n_test=3):
        images = []
        for _ in range(n_test):
            img = i2i(
                self.model,
                self.args.size,
                t,
                self.uc,
                self.sampler,
                ip=ip,
                step=100,
                scale=5,
                batch_size=self.batch_size,
                ddim_eta=0.0,
                dtype=self.dtype,
                device=self.device,
                camera=self.camera,
                num_frames=args.num_frames,
                pixel_control=(args.mode == "pixel"),
                transform=self.image_transform
            )
            img = np.concatenate(img, 1)
            images.append(img)
        return images

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        x = x.numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sd-v2.1-base-4view-ipmv",
        help="load pre-trained model from hugginface",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="load model from local config (override model_name)",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="path to local checkpoint"
    )
    parser.add_argument("--text", type=str, default="an astronaut riding a horse")
    parser.add_argument("--image", type=str, default="./assets/astrounaut.png")
    parser.add_argument("--name", type=str, default="yoda")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--num_frames", type=int, default=5, help=" \
        num of frames (views) to generate, should be in [4 or 5],  \
        5 for pixel control, 4 for local control"
    )
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=5)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--mode", type=str, default="pixel",
        help="ip mode default pixel"
    )
    args = parser.parse_args()

    t = args.text

    image_dream = ImageDreamDiffusion(args)
    assert args.num_frames in [4, 5], "num_frames should be in [4, 5]"
    assert os.path.exists(args.image), "image does not exist!"
    ip = []
    for i in range(8):
        img_idx = int(25 / 8 * i)
        img_path = os.path.join(args.image, '{0}.png'.format(img_idx))
        img1 = Image.open(img_path)
        img1 = add_random_background(img1, bg_color=255).resize((256, 256))
        ip.append(np.array(img1))
    ip_ = torch.from_numpy(np.array(ip))
    batch_size = max(4, args.num_frames) * 8
    device = args.device

    camera = get_camera(
        num_frames=4,
        elevation=0,
        azimuth_start=90,
        azimuth_span=360,
        extra_view=(args.mode == "pixel")
    )
    camera = camera.repeat(batch_size // args.num_frames, 1).to(device)
    image_dream.camera = camera
    images = image_dream.diffuse(t, ip_, n_test=1)

    images = np.concatenate(images, 0)

    test_save_path = './threestudio/models/imagedream/4dm/{0}'.format(args.name)
    for i in range(4):
        # out_image = []
        out_image_path = os.path.join(test_save_path, str(i))
        os.makedirs(out_image_path, exist_ok=True)
        for j in range(8):
            out_image = images[:, 256 * 5 * j + 256 * i: 256 * 5 * j + 256 * i + 256, :]
            Image.fromarray(out_image).save(f"{out_image_path}/{j}.png")
