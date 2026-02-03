import numpy as np
import torch
import torch.nn as nn
import logging
from imagenet_utils import DataSaverHook, StopForwardException
from qdrop.quantization.quantized_module import QuantizedModule
from qdrop.quantization.fake_quant import LSQFakeQuantize, LSQPlusFakeQuantize, QuantizeBase
from torch.nn import Module
from typing import List, Optional, Tuple, Union, Callable
from torch import Tensor
logger = logging.getLogger('qdrop')


def save_inp_oup_data(model: Module, module: Module, cali_data: Tensor,
                      store_inp: bool = False, store_oup: bool = False,
                      bs: int = 32, keep_gpu: bool = True) -> Tuple[Tuple[Optional[Tensor], ...], Union[Tensor, List]]: # 返回类型提示修正
    device = next(model.parameters()).device
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)

    cached_inputs = None # 使用列表的列表缓存输入
    cached_outputs = []

    with torch.no_grad():
        # 确定迭代批次数，处理末尾不足 bs 的情况
        num_samples = cali_data.size(0)
        num_batches = (num_samples + bs - 1) // bs # 向上取整

        for i in range(num_batches):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, num_samples)
            if start_idx >= end_idx: continue
            
            current_batch = cali_data[start_idx : end_idx].to(device)
            try:
                _ = model(current_batch)
            except StopForwardException:
                pass
            except Exception as e:
                 logger.warning(f"模型前向传播出错于批次 {i} (save_inp_oup_data): {e}")
                 continue

            if store_inp and data_saver.input_store is not None:
                input_tuple = data_saver.input_store
                if cached_inputs is None:
                    # 基于第一个有效批次的输入元组初始化列表的列表
                    cached_inputs = [[] for _ in input_tuple]

                # 确保 cached_inputs 列表数量与当前批次输入元组长度一致 (理论上应一致)
                if len(cached_inputs) == len(input_tuple):
                    for j, inp_tensor in enumerate(input_tuple):
                         if isinstance(inp_tensor, torch.Tensor):
                             current_inp_detached = inp_tensor.detach()
                             if not keep_gpu:
                                 current_inp_detached = current_inp_detached.cpu()
                             cached_inputs[j].append(current_inp_detached)
                         # else: 如果不是 Tensor (例如 None), 则不添加到对应的 inp_list 中
                # else: 如果输入元组长度变化，打印警告或抛出错误

            if store_oup and data_saver.output_store is not None:
                 output_tensor = data_saver.output_store
                 current_out_detached = output_tensor.detach()
                 if not keep_gpu:
                     current_out_detached = current_out_detached.cpu()
                 cached_outputs.append(current_out_detached)


    # --- 聚合输入 (从列表的列表) ---
    aggregated_inputs = []
    if store_inp and cached_inputs is not None:
        for inp_list in cached_inputs: # 遍历每个输入位置对应的列表
             if inp_list: # 如果这个位置收集到了 Tensor
                  aggregated_inputs.append(torch.cat(inp_list, dim=0))
             else:
                  # 如果某个输入位置从未收集到有效 Tensor，添加 None 占位
                  aggregated_inputs.append(None)
    return_inputs_tuple = tuple(aggregated_inputs) # 最终转换为元组
    # --- 结束输入聚合 ---

    return_outputs = []
    if store_oup and cached_outputs:
         return_outputs = torch.cat(cached_outputs, dim=0)

    handle.remove()
    torch.cuda.empty_cache()

    return return_inputs_tuple, return_outputs

class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 module: QuantizedModule,
                 weight: float = 1.,
                 iters: int = 20000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.):

        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(iters, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
        else:
            round_loss = 0
            for layer in self.module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    round_vals = layer.weight_fake_quant.rectified_sigmoid()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            logger.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


def lp_loss(pred, tgt, p: float = 2.0):
    loss_elementwise = (pred - tgt).abs().pow(p)

    if pred.ndim == 4:
        channel_dim = 1
    elif pred.ndim == 5:
        channel_dim = 2
    else:
        raise ValueError(f"lp_loss 函数期望输入为 4D 或 5D 张量, 但收到了 {pred.ndim}D 张量")

    loss = loss_elementwise.sum(dim=channel_dim).mean()
    return loss

def reconstruction(model, fp_model, module, fp_module, cali_data, config):
    device = next(module.parameters()).device
    # get data first
    quant_inp, _ = save_inp_oup_data(model, module, cali_data, store_inp=True, store_oup=False, bs=config.batch_size, keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data(fp_model, fp_module, cali_data, store_inp=True, store_oup=True, bs=config.batch_size, keep_gpu=config.keep_gpu)
    # prepare for up or down tuning
    if not isinstance(quant_inp, tuple) or not quant_inp: # 检查是否为元组且非空
        logger.warning(f"模块 {type(module).__name__} 未能获取有效的量化输入校准数据 (quant_inp 为空)，跳过重构。")
        return
    # --- 结束修改点 ---
    w_para, a_para = [], []
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = config.drop_prob
            if isinstance(layer, LSQFakeQuantize):
                a_para += [layer.scale]
            if isinstance(layer, LSQPlusFakeQuantize):
                a_para += [layer.scale]
                a_para += [layer.zero_point]
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None
    w_opt = torch.optim.Adam(w_para)
    loss_func = LossFunction(module=module, weight=config.weight, iters=config.iters, b_range=config.b_range,
                             warm_up=config.warm_up)

    sz = quant_inp[0].size(0)
    for i in range(config.iters):
        idx = torch.randint(0, sz, (config.batch_size,))
        cur_quant_inp_tuple = tuple(inp[idx].to(device) if inp is not None else None for inp in quant_inp)
        cur_fp_inp_tuple = tuple(inp[idx].to(device) if inp is not None else None for inp in fp_inp)
        cur_inp = []
        apply_drop = hasattr(config, 'drop_prob') and config.drop_prob < 1.0
        for q_in, f_in in zip(cur_quant_inp_tuple, cur_fp_inp_tuple):
            if q_in is None:
                 cur_inp.append(None)
            elif apply_drop and f_in is not None and q_in.shape == f_in.shape:
                 cur_inp.append(torch.where(torch.rand_like(q_in) < config.drop_prob, q_in, f_in))
            else: # 不应用 drop 或 fp 输入无效
                 cur_inp.append(q_in)
        cur_fp_oup = fp_oup[idx].to(device)
        if a_opt:
            a_opt.zero_grad()
        w_opt.zero_grad()
        cur_quant_oup = module(*cur_inp)
        if isinstance(cur_quant_oup, (tuple, list)):
            cur_quant_oup = cur_quant_oup[0]
        err = loss_func(cur_quant_oup, cur_fp_oup)
        err.backward()
        w_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()
    torch.cuda.empty_cache()
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0



