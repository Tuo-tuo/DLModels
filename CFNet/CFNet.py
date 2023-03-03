import torch
import torch.nn as nn

from torch import Tensor
from torchsummary import summary
from typing import Type, Any, Callable, Union, List, Optional, Dict



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CFNetParas(object):
    ###
    # Process Input Data
    ###
    input_h: int = 256
    input_w: int = 256
    input_channel: int = 3


    ###
    # Stem
    ###
    stem_kernel: int = 3
    stem_stride: int = 2
    stem_pad: int = 1
    stem_bias: bool = True
    stem_norm: nn.Module = nn.LayerNorm
    stem_acti: nn.Module = nn.GELU

    stem_conv2d_num: int = 2
    stem_output_channel: int = 64 # Can be changed.
    stem_channels: list = [input_channel] + [stem_output_channel]*stem_conv2d_num

    stem_output_shapes: list = list() # [B, C, H, W] -> [ [B, C, H/2, W/2], [B, C, H/4, W/4] ]
    for idx in range(stem_conv2d_num):
        stem_output_shapes.append([stem_channels[idx+1], input_h//(2**(idx+1)), input_w//(2**(idx+1))])
    

    ###
    # Block
    ###
    block_output_shape: list = stem_output_shapes[-1] # [B, C, H/4, W/4]
    block_in_channels: int = stem_output_shapes[-1][0] # C
    block_num: int = 1 # Can be changed.
    block_type: str = 'BTNK2'

    ###
    # Downsample out of stages.
    ###
    downsample_in_channels: int = block_in_channels # C
    downsample_output_shape: list = [block_output_shape[0], block_output_shape[1]//2, block_output_shape[2]//2] # [B, C, H/8, W/8]

    ###
    # Stage
    ###
    stage_num: int = 3

    # Block
    s1_b1_in_channels: int = downsample_in_channels
    s1_b1_out_channels: int = s1_b1_in_channels

    s1_ds1_in_channels: int = s1_b1_out_channels
    s1_ds1_out_channels: int = s1_ds1_in_channels # Can be changed

    s1_b2_in_channels: int = s1_ds1_out_channels
    s1_b2_out_channels: int = s1_b2_in_channels

    s1_ds2_in_channels: int = s1_b2_out_channels
    s1_ds2_out_channels: int = s1_ds2_in_channels # Can be changed

    # Focal Block
    s1_fb_in_channels: int = s1_ds2_out_channels
    s1_fb_out_channels: int = s1_fb_in_channels
    s1_fb_in_shape: List[int] = [s1_ds2_out_channels, downsample_output_shape[1]//4, downsample_output_shape[2]//4]
    s1_dilation_rate: int = 3
    s1_padding: int = 9
    
    # Transition Block
    s1_tb_in_shape: List[int] = s1_fb_in_shape
    s1_tb_c3_channels: int = s1_b1_out_channels
    s1_tb_c4_channels: int = s1_b2_out_channels
    s1_tb_c5_channels: int = s1_fb_out_channels

    stage_paras = {
                    'b_type':'BTNK2', 
                    'n1':1, 'b1_in_channels':s1_b1_in_channels, 'b1_out_channels':s1_b1_out_channels, 
                    'ds1_in_channels':s1_ds1_in_channels, 'ds1_out_channels':s1_ds1_out_channels, 
                    'n2':1, 'b2_in_channels':s1_b2_in_channels, 'b2_out_channels':s1_b2_out_channels, 
                    'ds2_in_channels':s1_ds2_in_channels, 'ds2_out_channels':s1_ds2_out_channels, 

                    'focal_block_type':'NeXt', 
                    'n3':1, 'fb_in_channels':s1_fb_in_channels, 'fb_out_channels':s1_fb_out_channels, 
                    'dilation_rate':s1_dilation_rate, 'padding':s1_padding,
                    'norm_layer':nn.LayerNorm, 'fb_in_shape':s1_fb_in_shape, 
                                     
                    'transition_block_type':'SeqAdd',
                    'c3_channels':s1_tb_c3_channels, 'c4_channels':s1_tb_c4_channels, 'c5_channels':s1_tb_c5_channels,
                    'is_final_stage':False
                }

    # Output
    o_in_channels = s1_tb_c3_channels
    o_output_channels = 1


class BTNK2(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.conv2d_1 = nn.Conv2d(in_channels, in_channels // 4, 1, 1)
        self.bn_1 = nn.BatchNorm2d(in_channels // 4)
        self.acti_1 = nn.ReLU(True)

        self.conv2d_2 = nn.Conv2d(in_channels // 4, in_channels // 4, 3, 1, padding=1)
        self.bn_2 = nn.BatchNorm2d(in_channels // 4)
        self.acti_2 = nn.ReLU(True)

        self.conv2d_3 = nn.Conv2d(in_channels // 4, in_channels, 1, 1)
        self.bn_3 = nn.BatchNorm2d(in_channels)

        self.acti_3 = nn.ReLU(True)
    
    def forward(self, x):
        identity = x

        out = self.conv2d_1(x)
        out = self.bn_1(out)
        out = self.acti_1(out)

        out = self.conv2d_2(out)
        out = self.bn_2(out)
        out = self.acti_2(out)

        out = self.conv2d_3(out)
        out = self.bn_3(out)
        
        out += identity
        out = self.acti_3(out)

        return out

class NeXt(nn.Module):
    def __init__(self, in_channels: int, dilation_rate: int, padding: int, norm_layer: Type[Callable[..., nn.Module]], in_shape: List[int]) -> None:
        super().__init__()

        self.d77 = self._depth_wise_convolution(in_channels, in_channels, 7, 1, 3, 1, 1, True, norm_layer, in_shape, nn.GELU)

        self.dd77 = self._depth_wise_convolution(in_channels, in_channels, 7, 1, padding, dilation_rate, in_channels, True, norm_layer, in_shape, nn.GELU)

        self.conv2d_1 = nn.Conv2d(in_channels, 4*in_channels, 1, 1)
        self.acti = nn.GELU()

        self.conv2d_2 = nn.Conv2d(4*in_channels, in_channels, 1, 1)
    
    def _depth_wise_convolution(self, in_channels, out_channels, 
                                kernel_size, stride, padding, dilation, groups, bias, 
                                norm_layer, norm_shape, acti_layer) -> nn.Sequential:
        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups, bias))
        layers.append(nn.Conv2d(in_channels, out_channels, 1, 1))

        layers.append(norm_layer(norm_shape))

        if acti_layer == nn.ReLU:
            layers.append(acti_layer(inplace=True))
        else:
            layers.append(acti_layer())
        
        return nn.Sequential(*layers)

    def forward(self, x):
        identity = x

        d77_out = self.d77(x)

        dd77_in = identity + d77_out
        dd77_out = self.dd77(dd77_in)

        conv2d_1_out = self.conv2d_1(dd77_out + dd77_in)

        conv2d_1_out = self.acti(conv2d_1_out)

        conv2d_2_out = self.conv2d_2(conv2d_1_out)

        return (conv2d_2_out + identity)

class SeqAdd(nn.Module):
    def __init__(self, c3_channels: int, c4_channels: int, c5_channels: int, is_final_stage: bool) -> None:
        super().__init__()

        self.conv2d_c4 = nn.Conv2d(c5_channels, c4_channels, 1, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv2d_c3 = nn.Conv2d(c4_channels, c3_channels, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.is_final_stage = is_final_stage
    
    def forward(self, c3, c4, c5):
        p5 = self.conv2d_c4(c5)
        p5 = self.upsample1(p5)

        p4 = self.conv2d_c3(p5 + c4)
        p4 = self.upsample2(p4)

        p3 = p4 + c3
        
        if False == self.is_final_stage:
            return p3
        else:
            return p3, p4, p5

class Stage(nn.Module):
    def __init__(self, stage_paras: Dict) -> None:
        super().__init__()

        self.block_1 = self._create_block(BTNK2, stage_paras['n1'], stage_paras['b1_in_channels'])
        self.downsample_1 = self._downsample_layer(stage_paras['ds1_in_channels'], stage_paras['ds1_out_channels'])

        self.block_2 = self._create_block(BTNK2, stage_paras['n2'], stage_paras['b2_in_channels'])
        self.downsample_2 = self._downsample_layer(stage_paras['ds2_in_channels'], stage_paras['ds2_out_channels'])

        self.focal_block = self._create_focal_block(NeXt, stage_paras['n3'], stage_paras['fb_in_channels'], stage_paras['dilation_rate'],
                                                    stage_paras['padding'], stage_paras['norm_layer'], stage_paras['fb_in_shape'])

        self.transition_block = self._create_transition_block(SeqAdd, stage_paras['c3_channels'], stage_paras['c4_channels'], 
                                                              stage_paras['c5_channels'], stage_paras['is_final_stage'])
        
        self.is_final_stage = stage_paras['is_final_stage']

        self.test_out_1 = nn.Conv2d(64, 1, 1)
        self.test_out_2 = nn.Conv2d(64, 1, 1)
        self.test_out_3 = nn.Conv2d(64, 1, 1)

    def _create_block(self, block: Type[BTNK2], block_num: int, block_in_channels: int) -> nn.Sequential:
        layers = []

        for _ in range(block_num):
            layers.append(block(block_in_channels))
        
        return nn.Sequential(*layers)

    def _downsample_layer(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels, 2, 2)
    
    def _create_focal_block(self, f_blcok: Type[NeXt], block_num: int, in_channels: int, 
                            dilation_rate: int, padding: int, 
                            norm_layer: Type[Callable[..., nn.Module]], in_shape: List[int]) -> nn.Sequential:
        layers = []

        for _ in range(block_num):
            layers.append(f_blcok(in_channels, dilation_rate, padding, norm_layer, in_shape))
        
        return nn.Sequential(*layers)

    def _create_transition_block(self, t_block: Type[SeqAdd], c3_channels: int, c4_channels: int, 
                                 c5_channels: int, is_final_stage: bool) -> nn.Module:
        return t_block(c3_channels, c4_channels, c5_channels, is_final_stage)

    def forward(self, x):
        c3 = self.block_1(x)
        downsample_1_out = self.downsample_1(c3)

        c4 = self.block_2(downsample_1_out)
        downsample_2_out = self.downsample_2(c4)

        c5 = self.focal_block(downsample_2_out)

        if False == self.is_final_stage:
            p3 = self.transition_block(c3, c4, c5)

            return p3
        else:
            p3, p4, p5 = self.transition_block(c3, c4, c5)

            # p3 = self.test_out_1(p3)
            # p4 = self.test_out_2(p4)
            # p5 = self.test_out_3(p5)
            return p3, p4, p5

class CFNet(nn.Module):
    def __init__(self, net_paras) -> None:
        super().__init__()
        self.net_paras = net_paras

        # [B, C, H, W] -> [B, C, H/4, W/4]
        self.stem = self._create_stem()

        # [B, C, H/4, W/4] -> [B, C, H/4, W/4]
        self.block = self._create_block(BTNK2, self.net_paras.block_num, self.net_paras.block_in_channels)

        # [B, C, H/4, W/4] -> [B, C, H/8, W/8]
        self.downsample_1 = self._downsample_layer(self.net_paras.downsample_in_channels, self.net_paras.downsample_in_channels)

        # [B, C, H/8, W/8] -> [B, C, H/8, W/8]
        self.stages = self._create_stages(Stage)

        # [B, C, H/8, W/8] -> [B, C, H/4, W/4]
        self.out = self._create_output(self.net_paras.o_in_channels, self.net_paras.o_output_channels)

    def _basic_conv2d(self, in_channels: int, out_channels: int, 
                            kernel_size: int, stride: int, padding: int, bias: bool, 
                            norm_layer: Type[Callable[..., nn.Module]], norm_shape: List[int], 
                            acti_layer: Type[Callable[..., nn.Module]]) -> nn.Sequential:
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        layers.append(norm_layer(norm_shape))
        if acti_layer == nn.ReLU:
            layers.append(acti_layer(inplace=True))
        else:
            layers.append(acti_layer())
        
        return nn.Sequential(*layers)

    def _create_stem(self) -> nn.Sequential:
        layers = []
        for idx in range(self.net_paras.stem_conv2d_num):
            layers.append(self._basic_conv2d(self.net_paras.stem_channels[idx], self.net_paras.stem_channels[idx+1], 
                                            self.net_paras.stem_kernel, self.net_paras.stem_stride, self.net_paras.stem_pad, self.net_paras.stem_bias,
                                            self.net_paras.stem_norm, self.net_paras.stem_output_shapes[idx], self.net_paras.stem_acti))
        
        return nn.Sequential(*layers)

    def _create_block(self, block: Type[BTNK2], block_num: int, block_in_channels: int) -> nn.Sequential:
        layers = []

        for _ in range(block_num):
            layers.append(block(block_in_channels))
        
        return nn.Sequential(*layers)

    def _downsample_layer(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels, 2, 2)

    def _create_stages(self, stage: Type[Stage]) -> nn.Sequential:
        stages = []
        for idx in range(self.net_paras.stage_num):
            stages.append(stage(self.net_paras.stage_paras))
        
        return nn.Sequential(*stages)

    def _create_output(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = []

        # [B, C, H/8, W/8] -> [B, C, H/4, W/4]
        layers.append(nn.Conv2d(in_channels, self.net_paras.stem_output_channel//2, 1))
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(self.net_paras.stem_output_channel//2))
        layers.append(nn.GELU())

        # [B, C, H/4, W/4] -> [B, C, H/2, W/2]
        layers.append(nn.Conv2d(self.net_paras.stem_output_channel//2, self.net_paras.stem_output_channel//4, 1))
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(self.net_paras.stem_output_channel//4))
        layers.append(nn.GELU())

        # [B, C, H/2, W/2] -> [B, C, H, W]
        layers.append(nn.Conv2d(self.net_paras.stem_output_channel//4, self.net_paras.stem_output_channel//8, 1))
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(self.net_paras.stem_output_channel//8))
        layers.append(nn.GELU())

        layers.append(nn.Conv2d(self.net_paras.stem_output_channel//8, out_channels, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(True))

        return nn.Sequential(*layers)


    def _forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.block(x)
        x = self.downsample_1(x)

        # p3, p4, p5 = self.stages(x)
        p3 = self.stages(x)

        out = self.out(p3)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)



if '__main__' == __name__:
    paras = CFNetParas()

    fc_net = CFNet(paras).to(device)

    summary(fc_net, (paras.input_channel, paras.input_h, paras.input_w), 4, device.type)
    

    inputs = torch.randint(0, 255, (4, paras.input_channel, paras.input_h, paras.input_w), dtype=torch.float32).to(device)
    fc_net_outputs = fc_net(inputs)

    print('\n ###################### fc_net_outputs.shape: ', fc_net_outputs.shape, '\n')

