# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
import numpy as np
from .utils import compute_scales
from .metrics import mse_loss

__all__ = ['OutlierSupressionPlusSearch']


class OutlierSupressionPlusSearch():
    def __init__(self, bits_length=8, loss_function=mse_loss):
        self.bnt = (1 << (bits_length - 1)) - 1
        self.loss_function = loss_function

    def _calculate_scales(self, act_abs_max, max_range=2.0):
        s = act_abs_max / max_range
        s = paddle.where(act_abs_max >= max_range, s, 1)
        return s

    def search(self, layer_name, act_input, act_abs_max, weight):
        act_max = float(act_abs_max.max())
        search_nums = max(100, int(act_max / 0.5))
        bounds = (1.0, act_max)
        step = (bounds[1] - bounds[0]) / search_nums
        act = act_input
        act.stop_gradient = True
        print('[smooth search] search input of %s' % layer_name)

        origin_out = paddle.matmul(act, weight)
        final_maxscale = bounds[1]
        search_threshold = bounds[1]
        calibration_loss = float('inf')
        w_abs_max = weight.abs().max(axis=-1, keepdim=True)
        smooth_scale_out = paddle.ones_like(act_abs_max)
        while search_threshold >= bounds[0]:
            s = self._calculate_scales(act_abs_max, search_threshold)

            new_act = act / s
            new_weight = weight * s.reshape(w_abs_max.shape)

            bnt = self.bnt
            quant_scale = compute_scales(new_act, method='abs_max')
            quant_act = paddle.clip(
                paddle.round(new_act / quant_scale * bnt), -bnt - 1, bnt)
            quant_dequant_act = quant_act / bnt * quant_scale

            quant_scale = compute_scales(
                new_weight, method='abs_max_channel_wise')
            quant_weight = paddle.clip(
                paddle.round(new_weight / quant_scale * bnt), -bnt - 1, bnt)
            quant_dequant_weight = quant_weight / bnt * quant_scale
            new_out = paddle.matmul(quant_dequant_act, quant_dequant_weight)

            cur_loss = self.loss_function(origin_out, new_out)
            if cur_loss <= calibration_loss:
                calibration_loss = cur_loss
                smooth_scale_out = s
                final_maxscale = search_threshold
            st -= step

        print("Layer {}, loss: {}, best search threshold: {}".format(
            layer_name, float(calibration_loss), float(final_maxscale)))
        return smooth_scale_out
