import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Type, List
from functools import partial
from torchsummary import summary
from timm.models.layers import trunc_normal_, DropPath

from custom_layers.norm_layers import LayerNorm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ConvNeXtParas(object):
    in_channels: int = 1
    in_height: int = 512
    in_width: int = 512
    block_num_per_stage: List[int] = [1, 1, 3, 1] # [3, 3, 9, 3]
    feature_dim_per_stage: List[int] = [96, 192, 384, 768] # [96, 192, 384, 768] [64, 128, 256, 512] [48, 96, 192, 384]
    downsample_conv_num: int = len(feature_dim_per_stage) - 1
    drop_path_rate: float = 0.
    layer_scale_init_value: float = 1e-6
    out_indices: List[int] = [0, 1, 2, 3] # Must be in ascending order


class FusionFeatureParas(object):
    feature_dims: List[int] = [ConvNeXtParas.feature_dim_per_stage[out_indice] for out_indice in ConvNeXtParas.out_indices[::-1]]
    feature_heights: List[int] = [ConvNeXtParas.in_height // (2 ** (2 + out_indice)) for out_indice in ConvNeXtParas.out_indices[::-1]]
    feature_widthes: List[int] = [ConvNeXtParas.in_width // (2 ** (2 + out_indice)) for out_indice in ConvNeXtParas.out_indices[::-1]]


class OutputParas(object):
    '''
    When upsampling feature, the number of feature channels doubles after each upsampling.
    When creating outputs, the number of feature channels halfs after each convolution, until it is equal to output_dims.
    '''
    input_dim: int = FusionFeatureParas.feature_dims[-1]
    feature_height: int = FusionFeatureParas.feature_heights[-1]
    feature_width: int = FusionFeatureParas.feature_widthes[-1]

    output_dims: int = 1

    upsample_num: int = int(math.log(ConvNeXtParas.in_height // feature_height, 2))
    assert upsample_num >= 1, 'upsample_num is less than 1.'

    upsample_dims: List[int] = [input_dim]
    for idx in range(upsample_num):
        upsample_dims.append(int(input_dim * (2 ** (idx+1))))
    

    out_conv_dim_factor: int = 2 # Can be changed.

    out_conv_num: int = int(math.log(upsample_dims[-1], out_conv_dim_factor))
    assert out_conv_num >= 1, 'out_conv_num is less than 1.'

    out_conv_dims: List[int] = [upsample_dims[-1]]
    for idx in range(out_conv_num):
        out_conv_dims.append(upsample_dims[-1] // (out_conv_dim_factor ** (idx+1)))

        if out_conv_dims[-1] < output_dims:
            out_conv_dims[-1] = output_dims * 2

    if out_conv_dims[-1] > output_dims:
        out_conv_dims[-1] = output_dims


class OutputParasV2(object):
    '''
    When upsampling feature, the number of feature channels halfs after each upsampling.
    When creating outputs, the number of feature channels halfs after each convolution, until it is equal to output_dims.
    '''
    input_dim: int = FusionFeatureParas.feature_dims[-1]
    feature_height: int = FusionFeatureParas.feature_heights[-1]
    feature_width: int = FusionFeatureParas.feature_widthes[-1]

    output_dims: int = 1

    upsample_num: int = int(math.log(ConvNeXtParas.in_height // feature_height, 2))
    assert upsample_num >= 1, 'upsample_num is less than 1.'

    upsample_dims: List[int] = [input_dim]
    for idx in range(upsample_num):
        upsample_dims.append(input_dim // (2 ** (idx+1)))
        
        if upsample_dims[-1] < 1:
            print('The upsample_num and upsample_dims is not suitable.')
            exit()
    

    out_conv_dim_factor: int = 2 # Can be changed.

    if upsample_dims[-1] > 1:
        out_conv_num: int = int(math.log(upsample_dims[-1], out_conv_dim_factor))
        assert out_conv_num >= 1, 'out_conv_num is less than 1.'

        out_conv_dims: List[int] = [upsample_dims[-1]]
        for idx in range(out_conv_num):
            out_conv_dims.append(upsample_dims[-1] // (out_conv_dim_factor ** (idx+1)))

            if out_conv_dims[-1] < output_dims:
                out_conv_dims[-1] = output_dims * 2

        if out_conv_dims[-1] > output_dims:
            out_conv_dims[-1] = output_dims
    else:
        out_conv_num: int = 1

        out_conv_dims: List[int] = [1, output_dims]


class OutputParasV3(object):
    '''
    When upsampling feature, the number of feature channels halfs after each upsampling.
    Only one convolution layer to creating outputs.
    '''
    input_dim: int = FusionFeatureParas.feature_dims[-1]
    feature_height: int = FusionFeatureParas.feature_heights[-1]
    feature_width: int = FusionFeatureParas.feature_widthes[-1]

    output_dims: int = 1

    upsample_num: int = int(math.log(ConvNeXtParas.in_height // feature_height, 2)) # Can be changed.

    upsample_dim_factor: int = 2 # Can be changed.

    upsample_dims: List[int] = [input_dim]
    for idx in range(upsample_num):
        upsample_dims.append(input_dim // ((2 ** (idx+1))))

        if upsample_dims[-1] < output_dims:
            upsample_dims[-1] = output_dims * 2


    out_conv_num: int = 1
    out_conv_dims: List[int] = [upsample_dims[-1], output_dims]


class ConvNeXtSegParas(object):
    ###
    # ConvNeXT
    ###
    convNeXtParas = ConvNeXtParas()

    ###
    # FusionFeature
    ###
    fusionFeatureParas = FusionFeatureParas()

    ###
    # Segmentation
    ###
    outputParas = OutputParas()
    

class Block(nn.Module):
    '''
    These codes refer to https://github.com/facebookresearch/ConvNeXt.

    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Parameters
        dim: int, Number of input channels.
        drop_path: float, Stochastic depth rate. Default: 0.0
        layer_scale_init_value: float, Init value for Layer Scale. Default: 1e-6.
    '''
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class BlockConvVer(nn.Module):
    '''
    These codes refer to https://github.com/facebookresearch/ConvNeXt.

    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Parameters
        dim: int, Number of input channels.
        drop_path: float, Stochastic depth rate. Default: 0.0
        layer_scale_init_value: float, Init value for Layer Scale. Default: 1e-6.
    '''
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, data_format = 'channels_first')
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1) # pointwise/1x1 convs, implemented with 1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    '''
    ConvNeXT, a pyTorch implement of 'A ConvNet for the 2020s', https://arxiv.org/pdf/2201.03545.pdf.
    These codes refer to https://github.com/facebookresearch/ConvNeXt.

    Parameters
        in_channels: int, number of input image channels.
        block_num_per_stage: list of int, number of blcoks at each stage.
        feature_dim_per_stage: list of int, feature dimension at each stage.
        downsample_conv_num: int, number of intermediate downsampling convolutional layer. Its value must be equal to (len(feature_dim_per_stage) - 1).
        drop_path_rate: float, stochastic depth rate.
        layer_scale_init_value: float, initial value for layer scale.
        out_indices: list of int, contain indexes of which outputs to be uesd to segmentation.
    '''
    def __init__(self, in_channels: int, block_num_per_stage: List[int], feature_dim_per_stage: List[int], downsample_conv_num: int, 
                 drop_path_rate: float, layer_scale_init_value: float, out_indices: List[int]) -> None:
        super().__init__()

        assert downsample_conv_num == (len(feature_dim_per_stage) - 1), 'The downsample_conv_num is not equal to (len(feature_dim_per_stage) - 1) !'

        self.block_num_per_stage = block_num_per_stage
        self.out_indices = out_indices

        '''
        Downsample Module
        '''
        self.downsample_module = nn.ModuleList() # Contain stem and 'downsample_conv_num' intermediate downsampling convolutional layers.

        stem = nn.Sequential(
                                nn.Conv2d(in_channels, feature_dim_per_stage[0], kernel_size=4, stride=4),
                                LayerNorm(feature_dim_per_stage[0], data_format='channels_first')
                            )
        self.downsample_module.append(stem)

        for idx in range(downsample_conv_num):
            downsample_layer = nn.Sequential(
                                                LayerNorm(feature_dim_per_stage[idx], data_format='channels_first'),
                                                nn.Conv2d(feature_dim_per_stage[idx], feature_dim_per_stage[idx+1], kernel_size=2, stride=2)
                                            )
            self.downsample_module.append(downsample_layer)
        
        '''
        Stage
        '''
        self.stages = nn.ModuleList() # len(feature_dim_per_stage) feature resolution stages, each consisting of multiple residual blocks.
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(block_num_per_stage))]
        cur_depth_idx = 0

        for idx in range(len(feature_dim_per_stage)):
            stage = nn.Sequential(
                                    *[Block(dim=feature_dim_per_stage[idx], drop_path=dp_rates[cur_depth_idx + depth_idx], 
                                    layer_scale_init_value=layer_scale_init_value) for depth_idx in range(block_num_per_stage[idx])]
                                )
            self.stages.append(stage)
            cur_depth_idx += block_num_per_stage[idx]
        
        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, data_format='channels_first')
        for idx in range(len(feature_dim_per_stage)):
            layer = norm_layer(feature_dim_per_stage[idx])
            layer_name = f'norm{idx}'
            self.add_module(layer_name, layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _forward(self, x):
        outs = []
        for idx in range(len(self.block_num_per_stage)):
            x = self.downsample_module[idx](x)
            x = self.stages[idx](x)

            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        
        return outs
        # return torch.

    def forward(self, x):
        x = self._forward(x)
        return x
    

class SeqAdd(nn.Module):
    '''
    PyTorch implement of Sequentially Add, comes from https://arxiv.org/abs/2302.06052.

    Parameters
        feature_dims, list of int, the dimensions of features outputed by ConvNeXt.
        feature_heights, list of int, the heights of features outputed by ConvNeXt.
        feature_widthes, list of int, the widthes of features outputed by ConvNeXt.
    '''
    def __init__(self, feature_dims, feature_heights, feature_widthes) -> Tensor:
        super().__init__()

        self.steps = len(feature_dims)-1
        self.conv2ds = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for idx in range(self.steps):
            self.conv2ds.append(nn.Conv2d(feature_dims[idx], feature_dims[idx+1], 1, 1))
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
    
    def forward(self, *args):
        for idx in range(self.steps):
            x = self.conv2ds[idx](args[self.steps - idx])
            x = self.upsamples[idx](x)
            x = x + args[self.steps - idx - 1]
        
        return x


class Outputs(nn.Module):
    '''
    This Outputs class is suitable for OutputParas class, OutputParasV2 class, and OutputParasV3 class.
    '''
    def __init__(self, paras: Type[OutputParas]):
        super().__init__()

        self.upsamples = nn.ModuleList()
        for idx in range(paras.upsample_num):
            upsample = nn.Sequential(
                                        nn.Conv2d(paras.upsample_dims[idx], paras.upsample_dims[idx+1], 1),
                                        # LayerNorm(paras.upsample_dims[idx+1], data_format='channels_first'),
                                        nn.BatchNorm2d(paras.upsample_dims[idx+1]),
                                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.GELU()
                                    )
            self.upsamples.append(upsample)
        

        self.out_convs = nn.ModuleList()
        for idx in range(paras.out_conv_num-1):
            out_conv = nn.Sequential(
                                        nn.Conv2d(paras.out_conv_dims[idx], paras.out_conv_dims[idx+1], 1),
                                        # LayerNorm(paras.out_conv_dims[idx+1], data_format='channels_first'),
                                        nn.BatchNorm2d(paras.out_conv_dims[idx+1]),
                                        nn.GELU()
                                    )
            self.out_convs.append(out_conv)
        out_conv = nn.Sequential(
                                        nn.Conv2d(paras.out_conv_dims[-2], paras.out_conv_dims[-1], 1),
                                        # LayerNorm(paras.out_conv_dims[-1], data_format='channels_first'),
                                        nn.BatchNorm2d(paras.out_conv_dims[-1]),
                                        # nn.ReLU(True)
                                        nn.Sigmoid()
                                )
        self.out_convs.append(out_conv)

    
    def forward(self, x):
        for ups in self.upsamples:
            x = ups(x)
        
        for out_conv in self.out_convs:
            x = out_conv(x)

        return x



class ConvNeXtSeg(nn.Module):
    def __init__(self, paras: Type[ConvNeXtSegParas]):
        super().__init__()

        self.cNextP = paras.convNeXtParas
        self.ffP = paras.fusionFeatureParas
        self.oP = paras.outputParas

        self.convNeXt = self._create_ConvNeXt(ConvNeXt, self.cNextP)

        self.ffeature = self._fusion_feature(SeqAdd, self.ffP)

        self.outputs = self._outputs(Outputs, self.oP)

    # ConvNeXT
    def _create_ConvNeXt(self, ds_module: Type[ConvNeXt], module_paras: Type[ConvNeXtParas]) -> nn.ModuleList:
        convNeXt = ds_module(module_paras.in_channels, module_paras.block_num_per_stage, module_paras.feature_dim_per_stage,
                            module_paras.downsample_conv_num, module_paras.drop_path_rate, module_paras.layer_scale_init_value,
                            module_paras.out_indices)

        return convNeXt
    
    # FusionFeature
    def _fusion_feature(self, ff_module: Type[SeqAdd], module_paras: Type[FusionFeatureParas]) -> nn.ModuleList:
        ffModule = ff_module(module_paras.feature_dims, module_paras.feature_heights, module_paras.feature_widthes)

        return ffModule
    
    # Segmentation
    def _outputs(self, o_module: Type[Outputs], module_paras: Type[OutputParas]) -> nn.ModuleList:
        oModule = o_module(module_paras)

        return oModule

    def _forward(self, x):
        x = self.convNeXt(x)

        x = self.ffeature(x[0], x[1], x[2], x[3])

        x = self.outputs(x)

        return x
    
    def forward(self, x):
        x = self._forward(x)
        return x


if '__main__' == __name__:
    '''
    Test ConvNeXt()
    '''
    # paras = ConvNeXtSegParas()

    # convNeXt = ConvNeXt(paras.convNeXtParas.in_channels, paras.convNeXtParas.block_num_per_stage, paras.convNeXtParas.feature_dim_per_stage,
    #                     paras.convNeXtParas.downsample_conv_num, paras.convNeXtParas.drop_path_rate, 
    #                     paras.convNeXtParas.layer_scale_init_value, paras.convNeXtParas.out_indices).to(device)
    
    # summary(convNeXt, (paras.convNeXtParas.in_channels, paras.convNeXtParas.in_height, paras.convNeXtParas.in_width), 4, device.type)

    # inputs = torch.randint(0, 255, (4, paras.convNeXtParas.in_channels, paras.convNeXtParas.in_height, paras.convNeXtParas.in_width), dtype=torch.float32).to(device)
    # convNeXt_outputs = convNeXt(inputs)

    # # print('\n ###################### convNeXt_outputs.shape: ', convNeXt_outputs.shape, '\n')
    # print('\n ###################### convNeXt_outputs.shape: \n')
    # for idx, convNeXt_output in enumerate(convNeXt_outputs):
    #     print(idx, ' ', convNeXt_output.shape)
    

    '''
    Test ConvNeXtSeg()
    '''
    paras = ConvNeXtSegParas()
    convNeXtSeg = ConvNeXtSeg(paras).to(device)
    summary(convNeXtSeg, (paras.convNeXtParas.in_channels, paras.convNeXtParas.in_height, paras.convNeXtParas.in_width), 4, device.type)


    inputs = torch.randint(0, 255, (2, paras.convNeXtParas.in_channels, paras.convNeXtParas.in_height, paras.convNeXtParas.in_width), dtype=torch.float32).to(device)
    convNeXtSeg_outputs = convNeXtSeg(inputs)

    # print('\n ###################### convNeXtSeg_outputs.shape: \n')
    # for idx, convNeXtSeg_output in enumerate(convNeXtSeg_outputs):
    #     print(idx, ' ', convNeXtSeg_output.shape)

    print('convNeXtSeg_outputs type and shape: ', type(convNeXtSeg_outputs), ' ', convNeXtSeg_outputs.shape)
