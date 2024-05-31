import rembg
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    default="yoda",
)
args = parser.parse_args()

name = args.name
bg_remover = rembg.new_session()
test_save_path = './threestudio/models/imagedream/4dm/{0}'.format(name)
for i in range(4):
    out_image_path = os.path.join(test_save_path, str(i))
    for j in range(8):
        file = f"{out_image_path}/{j}.png"
        file_out = f"{out_image_path}/{j}_rgba.png"

        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            img = rembg.remove(img, session=bg_remover)
            cv2.imwrite(file_out, img)
        else:
            cv2.imwrite(file_out, img)

reference_video_path = './threestudio/models/imagedream/assets/{0}'.format(name)
imgs = os.listdir(reference_video_path)
for img_path in imgs:
    if not img_path.endswith('.png'): continue
    img_num = img_path.split('.')[0]
    file = f"{reference_video_path}/{img_num}.png"
    file_out = f"{reference_video_path}/{img_num}_rgba.png"
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 3:
        img = rembg.remove(img, session=bg_remover)
        cv2.imwrite(file_out, img)
    else:
        cv2.imwrite(file_out, img)
