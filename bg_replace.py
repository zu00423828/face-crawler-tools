# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from glob import glob
import os
import sys
import cv2
import numpy as np
import paddle
from paddleseg.cvlibs import Config
from paddleseg.utils import get_sys_env
from tqdm import tqdm
import multiprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        description='PP-HumanSeg inference for video')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default='Matting/configs/modnet/modnet_mobilenetv2.yml',
        type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default='Matting/modnet-mobilenetv2.pdparams')
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='Image including human',
        type=str,
        default=None)
    parser.add_argument(
        '--trimap_path',
        dest='trimap_path',
        help='The path of trimap',
        type=str,
        default=None)
    parser.add_argument(
        '--bg_path',
        dest='bg_path',
        help='Background image path for replacing. If not specified, a white background is used',
        type=str,
        default=None)
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output')

    return parser.parse_args()


def main(args):
    sys.path.append('Matting')
    from core import predict
    import model
    from dataset import MattingDataset
    from utils import get_image_list
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'
    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    # elif len(val_dataset) == 0:
    #     raise ValueError(
    #         'The length of val_dataset is 0. Please check if your dataset is valid'
    #     )

    # msg = '\n---------------Config Information---------------\n'
    # msg += str(cfg)
    # msg += '------------------------------------------------'
    # logger.info(msg)

    model = cfg.model
    transforms = val_dataset.transforms

    # alpha = predict(
    #     model,
    #     model_path=args.model_path,
    #     transforms=transforms,
    #     image_list=[args.image_path],
    #     trimap_list=[args.trimap_path],
    #     save_dir=args.save_dir)
    if os.path.isdir(args.image_path):
        img_list = sorted(glob(f'{args.image_path}/*.png'))
        alpha = predict(
            model,
            model_path=args.model_path,
            transforms=transforms,
            image_list=img_list,
            save_dir=args.output_dir)
        with multiprocessing.Pool(4) as pool:
            for _ in tqdm(pool.imap_unordered(matting_img, img_list), total=len(img_list)):
                pass
    else:
        alpha = predict(
            model,
            model_path=args.model_path,
            transforms=transforms,
            image_list=[args.image_path],
            trimap_list=[args.trimap_path],
            save_dir=args.output_dir)

        img_ori = cv2.imread(args.image_path)
        bg = get_bg(args.bg_path, img_ori.shape)
        alpha = alpha / 255
        alpha = alpha[:, :, np.newaxis]
        com = alpha * img_ori + (1 - alpha) * bg
        com = com.astype('uint8')
        com_save_path = os.path.join(args.output_dir,
                                     os.path.basename(args.image_path))
        cv2.imwrite(com_save_path, com)


def matting_img(image_path):
    img_ori = cv2.imread(image_path)
    alpha_path = os.path.join(
        args.output_dir, os.path.basename(image_path))
    alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
    bg = get_bg(args.bg_path, img_ori.shape)
    alpha = alpha / 255
    alpha = alpha[:, :, np.newaxis]
    com = alpha * img_ori + (1 - alpha) * bg
    com = com.astype('uint8')
    com_save_path = os.path.join(args.output_dir,
                                 os.path.basename(image_path))
    cv2.imwrite(com_save_path, com)
    return None


def get_bg(bg_path, img_shape):
    if bg_path is None:
        bg = np.ones(img_shape)
        bg *= 255

    elif not os.path.exists(bg_path):
        raise Exception('The --bg_path is not existed: {}'.format(bg_path))
    else:
        bg = cv2.imread(bg_path)
        bg = cv2.resize(bg, (img_shape[1], img_shape[0]))
    return bg


if __name__ == "__main__":
    args = parse_args()
    main(args)
