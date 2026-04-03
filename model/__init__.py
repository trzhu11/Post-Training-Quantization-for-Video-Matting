import torch.nn as nn
import torch
from quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer   # noqa: F401
from typing import Optional
from torch import Tensor
from .mobilenetv3 import InvertedResidual
from torchvision.ops.misc import SqueezeExcitation
from .lraspp import LRASPP
from .decoder import ConvGRU, BottleneckBlock, UpsamplingBlock, OutputBlock, Projection
from .model import MattingNetwork
from .deep_guided_filter import DeepGuidedFilterRefiner
from torch.nn import functional as F
class QuantSqueezeExcitation(nn.Module):
    def __init__(self, org_module: SqueezeExcitation, w_qconfig, a_qconfig):
        super().__init__()
        self.avgpool = org_module.avgpool

        fc1_conv = org_module.fc1
        activation = org_module.activation
        self.quant_fc1 = QuantizedLayer(
            module=fc1_conv,
            activation=activation,
            w_qconfig=w_qconfig,
            a_qconfig=a_qconfig,
            qoutput=True # SE内部的中间层, 输出需要量化
        )

        fc2_conv = org_module.fc2
        scale_activation = org_module.scale_activation
        self.quant_fc2 = QuantizedLayer(
            module=fc2_conv,
            activation=scale_activation,
            w_qconfig=w_qconfig,
            a_qconfig=a_qconfig,
            qoutput=True # SE最终输出在外部与其他特征相乘, 这里量化是合理的
        )

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.quant_fc1(scale)
        scale = self.quant_fc2(scale)
        return scale

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input

class QuantInvertedResidual_RVM(QuantizedBlock):
    def __init__(self, org_module: InvertedResidual, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.use_res_connect = org_module.use_res_connect
        self.use_se = org_module.use_se

        current_idx = 0 # 使用动态索引计数器

        self.ql_expand = None
        if org_module.expanded:
            expand_seq = org_module.block[current_idx] # 使用当前索引访问 expand 层
            expand_conv = expand_seq[0]
            expand_act = expand_seq[2]
            self.ql_expand = QuantizedLayer(
                module=expand_conv,
                activation=expand_act,
                w_qconfig=w_qconfig,
                a_qconfig=a_qconfig
            )
            current_idx += 1 # 访问后增加索引

        # 访问 depthwise 层
        dw_seq = org_module.block[current_idx]
        dw_conv = dw_seq[0]
        dw_act = dw_seq[2]
        self.ql_dw = QuantizedLayer(
            module=dw_conv,
            activation=dw_act,
            w_qconfig=w_qconfig,
            a_qconfig=a_qconfig
        )
        current_idx += 1 # 访问后增加索引

        self.se_layer_quant = None
        if self.use_se:
            org_se_layer = org_module.block[current_idx]
            # 实例化 QuantSqueezeExcitation 替换原始 SE 层
            self.se_layer_quant = QuantSqueezeExcitation(
                org_module=org_se_layer,
                w_qconfig=w_qconfig,
                a_qconfig=a_qconfig
            )
            current_idx += 1
        # 访问 project 层
        proj_seq = org_module.block[current_idx]
        proj_conv = proj_seq[0]
        self.ql_proj = QuantizedLayer(
            module=proj_conv,
            activation=None,
            w_qconfig=w_qconfig,
            a_qconfig=a_qconfig,
            qoutput=False
        )
        # Project 层是最后一部分，不需要再增加 current_idx

        self.block_post_act_fake_quantize = None
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, config=a_qconfig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.ql_expand is not None:
            out = self.ql_expand(x)
        else:
            out = x

        out = self.ql_dw(out)

        if self.se_layer_quant is not None:
            out = self.se_layer_quant(out)

        out = self.ql_proj(out)

        if self.use_res_connect:
            out = out + identity

        if self.block_post_act_fake_quantize is not None:
            out = self.block_post_act_fake_quantize(out)

        return out

class QuantLRASPP(QuantizedBlock):
    def __init__(self, org_module: LRASPP, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput

        conv1 = org_module.aspp1[0]
        relu1 = org_module.aspp1[2]
        self.quant_aspp1 = QuantizedLayer(
            module=conv1,
            activation=relu1,
            w_qconfig=w_qconfig,
            a_qconfig=a_qconfig
        )

        self.quant_pool2 = org_module.aspp2[0]
        conv2 = org_module.aspp2[1]
        sigmoid2 = org_module.aspp2[2]
        self.quant_aspp2_conv = QuantizedLayer(
            module=conv2,
            activation=sigmoid2,
            w_qconfig=w_qconfig,
            a_qconfig=a_qconfig
        )

        self.block_post_act_fake_quantize = None
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, config=a_qconfig)

    def forward_single_frame(self, x: Tensor) -> Tensor:
        y1 = self.quant_aspp1(x)
        y2 = self.quant_pool2(x)
        y2 = self.quant_aspp2_conv(y2)
        out = y1 * y2
        return out

    def forward_time_series(self, x: Tensor) -> Tensor:
        B, T = x.shape[:2]
        x_flat = x.flatten(0, 1)
        y_flat = self.forward_single_frame(x_flat)
        y = y_flat.unflatten(0, (B, T))
        return y

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            out = self.forward_time_series(x)
        else:
            out = self.forward_single_frame(x)

        if self.block_post_act_fake_quantize is not None:
            out = self.block_post_act_fake_quantize(out)

        return out

class QuantConvGRU(nn.Module):
    def __init__(self, org_module: ConvGRU, w_qconfig, a_qconfig):
        super().__init__()
        self.channels = org_module.channels

        ih_conv = org_module.ih[0]
        ih_act = org_module.ih[1]
        self.quant_ih = QuantizedLayer(
            module=ih_conv,
            activation=ih_act,
            w_qconfig=w_qconfig,
            a_qconfig=a_qconfig
        )

        hh_conv = org_module.hh[0]
        hh_act = org_module.hh[1]
        self.quant_hh = QuantizedLayer(
            module=hh_conv,
            activation=hh_act,
            w_qconfig=w_qconfig,
            a_qconfig=a_qconfig
        )

    def forward_single_frame(self, x, h):
        if h is None: h = torch.zeros((x.size(0), self.channels, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        r, z = self.quant_ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.quant_hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h):
        if h is None: h = torch.zeros((x.size(0), self.channels, x.size(-2), x.size(-1)), device=x.device, dtype=x.dtype)
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h

    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h_shape = (x.size(0), self.channels, x.size(-2), x.size(-1))
            h = torch.zeros(h_shape, device=x.device, dtype=x.dtype)

        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class QuantBottleneckBlock(QuantizedBlock):
    def __init__(self, org_module: BottleneckBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.channels = org_module.channels
        self.gru = QuantConvGRU(org_module.gru, w_qconfig, a_qconfig)
        self.block_post_act_fake_quantize = None
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, config=a_qconfig)

    def forward(self, x, r: Optional[Tensor]):
        channel_dim = 1 if x.ndim == 4 else 2
        a, b = x.split(self.channels // 2, dim=channel_dim)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=channel_dim)

        if self.block_post_act_fake_quantize is not None:
            x = self.block_post_act_fake_quantize(x)
        return x, r


class QuantUpsamplingBlock(QuantizedBlock):
    def __init__(self, org_module: UpsamplingBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.out_channels = org_module.out_channels
        self.upsample = org_module.upsample

        conv_seq = org_module.conv
        self.quant_conv = QuantizedLayer(
                module=conv_seq[0], activation=conv_seq[2],
                w_qconfig=w_qconfig, a_qconfig=a_qconfig
            )
        self.gru = QuantConvGRU(org_module.gru, w_qconfig, a_qconfig)
        self.block_post_act_fake_quantize = None
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, config=a_qconfig)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.quant_conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r

    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s], dim=1)
        x = self.quant_conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)
        return x, r

    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            out, r_out = self.forward_time_series(x, f, s, r)
        else:
            out, r_out = self.forward_single_frame(x, f, s, r)

        if self.block_post_act_fake_quantize is not None:
            out = self.block_post_act_fake_quantize(out)
        return out, r_out


class QuantOutputBlock(QuantizedBlock):
    def __init__(self, org_module: OutputBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.upsample = org_module.upsample

        conv_seq = org_module.conv
        quant_conv_layers = []
        quant_conv_layers.append(QuantizedLayer(
            module=conv_seq[0], activation=conv_seq[2],
            w_qconfig=w_qconfig, a_qconfig=a_qconfig
        ))
        quant_conv_layers.append(QuantizedLayer(
            module=conv_seq[3], activation=conv_seq[5],
            w_qconfig=w_qconfig, a_qconfig=a_qconfig, qoutput=False
        ))
        self.quant_conv = nn.Sequential(*quant_conv_layers)

        self.block_post_act_fake_quantize = None
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, config=a_qconfig)

    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.quant_conv(x)
        return x

    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.quant_conv(x)
        x = x.unflatten(0, (B, T))
        return x

    def forward(self, x, s):
        if x.ndim == 5:
            out = self.forward_time_series(x, s)
        else:
            out = self.forward_single_frame(x, s)

        if self.block_post_act_fake_quantize is not None:
            out = self.block_post_act_fake_quantize(out)
        return out


class QuantProjection(QuantizedBlock):
    def __init__(self, org_module: Projection, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput

        conv = org_module.conv
        self.quant_conv = QuantizedLayer(
            module=conv,
            activation=None,
            w_qconfig=w_qconfig,
            a_qconfig=a_qconfig,
            qoutput=False
        )

        self.block_post_act_fake_quantize = None
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, config=a_qconfig)

    def forward_single_frame(self, x: Tensor) -> Tensor:
        return self.quant_conv(x)

    def forward_time_series(self, x: Tensor) -> Tensor:
        B, T = x.shape[:2]
        return self.quant_conv(x.flatten(0, 1)).unflatten(0, (B, T))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            out = self.forward_time_series(x)
        else:
            out = self.forward_single_frame(x)

        if self.block_post_act_fake_quantize is not None:
            out = self.block_post_act_fake_quantize(out)

        return out

class QuantDeepGuidedFilterRefiner(QuantizedBlock):
    def __init__(self, org_module: DeepGuidedFilterRefiner, w_qconfig, a_qconfig, qoutput=False):
        super().__init__()
        self.box_filter = QuantizedLayer(org_module.box_filter, None, w_qconfig, a_qconfig, False)
        self.conv = nn.Sequential(
            QuantizedLayer(org_module.conv[0], org_module.conv[2], w_qconfig, a_qconfig),
            QuantizedLayer(org_module.conv[3], org_module.conv[5], w_qconfig, a_qconfig),
            QuantizedLayer(org_module.conv[6], None, w_qconfig, a_qconfig, False)
        )
        self.block_post_act_fake_quantize = None
        
    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        fine_x = torch.cat([fine_src, fine_src.mean(1, keepdim=True)], dim=1)
        base_x = torch.cat([base_src, base_src.mean(1, keepdim=True)], dim=1)
        base_y = torch.cat([base_fgr, base_pha], dim=1)
        
        mean_x = self.box_filter(base_x)
        mean_y = self.box_filter(base_y)
        cov_xy = self.box_filter(base_x * base_y) - mean_x * mean_y
        var_x  = self.box_filter(base_x * base_x) - mean_x * mean_x
        
        A = self.conv(torch.cat([cov_xy, var_x, base_hid], dim=1))
        b = mean_y - A * mean_x
        
        H, W = fine_src.shape[2:]
        A = F.interpolate(A, (H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, (H, W), mode='bilinear', align_corners=False)
        
        out = A * fine_x + b
        fgr, pha = out.split([3, 1], dim=1)
        return fgr, pha
    
    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(
            fine_src.flatten(0, 1),
            base_src.flatten(0, 1),
            base_fgr.flatten(0, 1),
            base_pha.flatten(0, 1),
            base_hid.flatten(0, 1))
        fgr = fgr.unflatten(0, (B, T))
        pha = pha.unflatten(0, (B, T))
        return fgr, pha
    
    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_fgr, base_pha, base_hid)
        else:
            return self.forward_single_frame(fine_src, base_src, base_fgr, base_pha, base_hid)

specials = {
    InvertedResidual: QuantInvertedResidual_RVM,
    LRASPP: QuantLRASPP,
    BottleneckBlock: QuantBottleneckBlock,
    UpsamplingBlock: QuantUpsamplingBlock,
    OutputBlock: QuantOutputBlock,
    Projection: QuantProjection
}


def load_model(config):
    config['kwargs'] = config.get('kwargs', dict())
    model = eval(config['type'])(**config['kwargs'])
    checkpoint = torch.load(config.path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    return model
