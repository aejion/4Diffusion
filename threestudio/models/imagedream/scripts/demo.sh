# Run this script under ImageDream/
export PYTHONPATH=$PYTHONPATH:./
#export HF_ENDPOINT=https://hf-mirror.com

# test pixel version
python3 threestudio/models/imagedream/scripts/demo.py  \
    --image "./threestudio/models/imagedream/assets/yoda" \
    --name "yoda" \
    --text "baby yoda in the style of Mormookiee" \
    --config_path "./threestudio/models/imagedream/imagedream/configs/sd_v2_base_ipmv.yaml" \
    --ckpt_path "./ckpts/sd-v2.1-base-4view-ipmv.pt" \
    --mode "pixel" \
    --num_frames 5
