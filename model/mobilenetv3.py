import torch
from torch import nn, Tensor
# from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidualConfig, InvertedResidual
from torchvision.transforms.functional import normalize
from torchvision.models._utils import _make_divisible
from functools import partial
from typing import Any, Callable, List, Optional, Sequence
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer

class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)
    
class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        self.norm_layer = norm_layer
        self.cnf = cnf
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        # Expand层
        self.expanded = False
        if cnf.expanded_channels != cnf.input_channels:
            self.expanded = True
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        cnf.input_channels,
                        cnf.expanded_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,  # kernel_size=1时padding自动为0
                        bias=norm_layer is None  # 如果无归一化层则启用bias
                    ),
                    norm_layer(cnf.expanded_channels),
                    activation_layer(inplace=False)  # 默认使用inplace操作
                )
            )

        # Depthwise层
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    cnf.expanded_channels,
                    cnf.expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=stride,
                    padding=(cnf.kernel - 1) // 2 * cnf.dilation,  # 根据kernel和dilation计算padding
                    dilation=cnf.dilation,
                    groups=cnf.expanded_channels,  # 深度可分离卷积的关键
                    bias=norm_layer is None
                ),
                norm_layer(cnf.expanded_channels),
                activation_layer(inplace=False)
            )
        )

        # SE模块（无需修改）
        self.use_se = cnf.use_se
        self.expanded_channels = cnf.expanded_channels

        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            self.squeeze_channels = squeeze_channels
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # Project层
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    cnf.expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=norm_layer is None
                ),
                norm_layer(cnf.out_channels),
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = result + input
        return result

class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
            
        layers: List[nn.Module] = []
        
        # 首层卷积替换
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, firstconv_output_channels, kernel_size=3, stride=2, 
                         padding=1,  # kernel_size=3时padding=1
                         bias=norm_layer is None),
                norm_layer(firstconv_output_channels),
                nn.Hardswish(inplace=False)
            )
        )

        # 倒残差块保持不变
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # 最后一层卷积替换
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(lastconv_input_channels, lastconv_output_channels, 
                         kernel_size=1, stride=1, padding=0,
                         bias=norm_layer is None),
                norm_layer(lastconv_output_channels),
                nn.Hardswish(inplace=False)
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=False),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(last_channel, num_classes),
        )

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class MobileNetV3Encoder(MobileNetV3):
    def __init__(self, pretrained: bool = False):
        super().__init__(
            inverted_residual_setting=[
                InvertedResidualConfig( 16, 3,  16,  16, False, "RE", 1, 1, 1),
                InvertedResidualConfig( 16, 3,  64,  24, False, "RE", 2, 1, 1),  # C1
                InvertedResidualConfig( 24, 3,  72,  24, False, "RE", 1, 1, 1),
                InvertedResidualConfig( 24, 5,  72,  40,  True, "RE", 2, 1, 1),  # C2
                InvertedResidualConfig( 40, 5, 120,  40,  True, "RE", 1, 1, 1),
                InvertedResidualConfig( 40, 5, 120,  40,  True, "RE", 1, 1, 1),
                InvertedResidualConfig( 40, 3, 240,  80, False, "HS", 2, 1, 1),  # C3
                InvertedResidualConfig( 80, 3, 200,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 184,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 184,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 480, 112,  True, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 3, 672, 112,  True, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 5, 672, 160,  True, "HS", 2, 2, 1),  # C4
                InvertedResidualConfig(160, 5, 960, 160,  True, "HS", 1, 2, 1),
                InvertedResidualConfig(160, 5, 960, 160,  True, "HS", 1, 2, 1),
            ],
            last_channel=1280
        )
        if pretrained:
            # 使用本地路径加载预训练模型权重
            self.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
        # if pretrained:
        #     self.load_state_dict(torch.hub.load_state_dict_from_url(
        #         'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'))

        del self.avgpool
        del self.classifier
        
    def forward_single_frame(self, x):
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        x = self.features[0](x)
        x = self.features[1](x)
        f1 = x
        x = self.features[2](x)
        x = self.features[3](x)
        f2 = x
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        f3 = x
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        f4 = x
        return [f1, f2, f3, f4]
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

