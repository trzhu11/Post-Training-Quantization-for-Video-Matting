import numpy as np  # noqa: F401
import copy
import time
import torch
import torch.nn as nn
import logging
import argparse
import videomatte_utils
from qdrop.solver.recon import reconstruction
from qdrop.solver.fold_bn import search_fold_and_remove_bn, StraightThrough
from qdrop.model import load_model, specials
from qdrop.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from qdrop.quantization.quantized_module import QuantizedLayer, QuantizedBlock
from qdrop.quantization.fake_quant import QuantizeBase
from qdrop.quantization.observer import ObserverBase
import datetime # 新增导入
import os # 新增导入
import yaml # 新增导入 (需要 pip install pyyaml)
logger = logging.getLogger('qdrop')
logging.basicConfig(level=logging.INFO, format='%(message)s')


def quantize_model(model, config_quant):

    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
        childs = list(iter(module.named_children()))
        st, ed = 0, len(childs)
        prev_quantmodule = None
        while(st < ed):
            tmp_qoutput = qoutput if st == ed - 1 else True
            name, child_module = childs[st][0], childs[st][1]
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, tmp_qoutput))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput=tmp_qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Hardswish, nn.Tanh )):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
            st += 1
    replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig, qoutput=False)
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'weight' in name:
            w_list.append(module)
        if isinstance(module, QuantizeBase) and 'act' in name:
            a_list.append(module)
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8)
    'the image input has already been in 256, set the last layer\'s input to 8-bit'
    a_list[-1].set_bit(8)
    logger.info('finish quantize model:\n{}'.format(str(model)))
    return model


def get_cali_data(train_loader, num_samples):
    fgr_samples, pha_samples, bgr_samples = [], [], []
    for batch in train_loader:
        true_fgr, true_pha, true_bgr = batch[0], batch[1], batch[2]
        fgr_samples.append(true_fgr)
        pha_samples.append(true_pha)
        bgr_samples.append(true_bgr)
        # 用任意一个tensor的size计算累计样本数
        if len(fgr_samples) * true_fgr.size(0) >= num_samples:
            break
    # 统一截取并返回三元组
    truncate = lambda x: torch.cat(x, dim=0)[:num_samples]
    return truncate(fgr_samples), truncate(pha_samples), truncate(bgr_samples)


def main(config_path):
    config = videomatte_utils.parse_config(config_path)
    gpu_id_to_use = config.gpu_id
    device = torch.device(f'cuda:{gpu_id_to_use}')
    torch.cuda.set_device(device)
    videomatte_utils.set_seed(config.process.seed)
    'cali data'
    train_loader = videomatte_utils.load_data(**config.data)
    cali_fgr, cali_pha, cali_bgr = get_cali_data(train_loader, config.quant.calibrate)
    cali_data = cali_fgr * cali_pha + cali_bgr * (1 - cali_pha)
    'model'
    model = load_model(config.model)
    search_fold_and_remove_bn(model)
    if hasattr(config, 'quant'):
        model = quantize_model(model, config.quant)
    model.cuda()
    model.eval()
    fp_model = copy.deepcopy(model)
    disable_all(fp_model)
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)

    # calibrate first
    with torch.no_grad():
        st = time.time()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        model(cali_data[: 1].cuda())
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[: 1].cuda())
        ed = time.time()
        logger.info('the calibration time is {}'.format(ed - st))

    if hasattr(config.quant, 'recon'):
        enable_quantization(model)

        def recon_model(module: nn.Module, fp_module: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                    logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                    reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data, config.quant.recon)
                else:
                    recon_model(child_module, getattr(fp_module, name))
        # Start reconstruction
        recon_model(model, fp_model)
    enable_quantization(model)


    # --- 修改后的模型和配置保存代码 ---
    try:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        save_dir = "/home/test/pytorch_ztr/QDrop/qdrop/saved_models/granularity/block/4-4"
        base_filename = f"quantized_rvm_model_{timestamp}"

        os.makedirs(save_dir, exist_ok=True)

        model_save_path = os.path.join(save_dir, f"{base_filename}.pth")
        config_save_path = os.path.join(save_dir, f"{base_filename}_config.yaml")

        # 1. 保存模型
        torch.save(model, model_save_path)
        logger.info(f"已将量化后的整个模型保存到: {model_save_path}")

        # 2. 保存配置
        try:
            config_to_save = None
            if isinstance(config, dict):
                config_to_save = config
            elif hasattr(config, '__dict__'): # 尝试转换 Namespace 或简单对象
                config_to_save = vars(config)
            # else: # 处理其他无法直接转换的 config 类型
            #     logger.warning("无法自动将 config 对象转换为字典，尝试直接保存...")
            #     config_to_save = config # 直接尝试保存原始对象 (可能失败)

            if config_to_save is not None:
                with open(config_save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                logger.info(f"已将配置文件保存到: {config_save_path}")
            else:
                logger.warning("无法将 config 对象转换为可保存的字典格式，跳过配置文件保存。")

        except Exception as e_cfg:
            logger.error(f"保存配置文件时出错: {e_cfg}")

    except Exception as e_model:
        logger.error(f"保存模型时出错: {e_model}")
    # --- 结束修改后的代码 ---



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
