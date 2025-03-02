# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import functools
from functools import partial
import math

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from imagenet_reader import ImageNetDataset
from paddleslim.common import load_config as load_slim_config
from paddleslim.auto_compression import AutoCompression


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help="directory to save compressed model.")
    parser.add_argument(
        '--total_images',
        type=int,
        default=1281167,
        help="the number of total training images.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    return parser


# yapf: enable
def reader_wrapper(reader, input_name):
    def gen():
        for i, (imgs, label) in enumerate(reader()):
            yield {input_name: imgs}

    return gen


def eval_reader(data_dir, batch_size, crop_size, resize_size, place=None):
    val_reader = ImageNetDataset(
        mode='val',
        data_dir=data_dir,
        crop_size=crop_size,
        resize_size=resize_size)
    val_loader = DataLoader(
        val_reader,
        places=[place] if place is not None else None,
        batch_size=global_config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=0)
    return val_loader


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    val_loader = eval_reader(
        data_dir,
        batch_size=global_config['batch_size'],
        crop_size=img_size,
        resize_size=resize_size)

    results = []
    print('Evaluating...')
    for batch_id, (image, label) in enumerate(val_loader):
        # top1_acc, top5_acc
        if len(test_feed_names) == 1:
            image = np.array(image)
            label = np.array(label).astype('int64')
            pred = exe.run(compiled_test_program,
                           feed={test_feed_names[0]: image},
                           fetch_list=test_fetch_list)
            pred = np.array(pred[0])
            label = np.array(label)
            sort_array = pred.argsort(axis=1)
            top_1_pred = sort_array[:, -1:][:, ::-1]
            top_1 = np.mean(label == top_1_pred)
            top_5_pred = sort_array[:, -5:][:, ::-1]
            acc_num = 0
            for i in range(len(label)):
                if label[i][0] in top_5_pred[i]:
                    acc_num += 1
            top_5 = float(acc_num) / len(label)
            results.append([top_1, top_5])
        else:
            # eval "eval model", which inputs are image and label, output is top1 and top5 accuracy
            image = np.array(image)
            label = np.array(label).astype('int64')
            result = exe.run(
                compiled_test_program,
                feed={test_feed_names[0]: image,
                      test_feed_names[1]: label},
                fetch_list=test_fetch_list)
            result = [np.mean(r) for r in result]
            results.append(result)
        if batch_id % 100 == 0:
            print('Eval iter: ', batch_id)
    result = np.mean(np.array(results), axis=0)
    return result[0]


def main():
    rank_id = paddle.distributed.get_rank()
    if args.devices == 'gpu':
        place = paddle.CUDAPlace(rank_id)
        paddle.set_device('gpu')
    else:
        place = paddle.CPUPlace()
        paddle.set_device('cpu')
    global global_config
    all_config = load_slim_config(args.config_path)

    assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
    global_config = all_config["Global"]

    gpu_num = paddle.distributed.get_world_size()
    if isinstance(all_config['TrainConfig']['learning_rate'],
                  dict) and all_config['TrainConfig']['learning_rate'][
                      'type'] == 'CosineAnnealingDecay':
        step = int(
            math.ceil(
                float(args.total_images) / (global_config['batch_size'] *
                                            gpu_num)))
        all_config['TrainConfig']['learning_rate']['T_max'] = step
        print('total training steps:', step)

    global data_dir
    data_dir = global_config['data_dir']

    global img_size, resize_size
    img_size = global_config['img_size'] if 'img_size' in global_config else 224
    resize_size = global_config[
        'resize_size'] if 'resize_size' in global_config else 256

    train_dataset = ImageNetDataset(
        mode='train',
        data_dir=data_dir,
        crop_size=img_size,
        resize_size=resize_size)

    train_loader = DataLoader(
        train_dataset,
        places=[place],
        batch_size=global_config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=0)
    train_dataloader = reader_wrapper(train_loader, global_config['input_name'])

    ac = AutoCompression(
        model_dir=global_config['model_dir'],
        model_filename=global_config['model_filename'],
        params_filename=global_config['params_filename'],
        save_dir=args.save_dir,
        config=all_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function if rank_id == 0 else None,
        eval_dataloader=reader_wrapper(
            eval_reader(
                data_dir,
                global_config['batch_size'],
                crop_size=img_size,
                resize_size=resize_size,
                place=place),
            global_config['input_name']))

    ac.compress()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main()
