import torch
import platform
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, List
from torchsummary import summary



if platform.system().lower() in ['windows', 'linux']:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')



class UnetParas(object):
    ''' Input data'''
    in_data_channel: int = 1
    in_data_height: int = 160
    in_data_width: int = 160

    ''' Downsample '''
    down_channels: List[int] = [32, 64, 128] # [64, 128, 256, 512] # The last channel the bottom channel.
    d_conv_num_per_step: int = 2
    
    ''' Upsample '''
    up_channels: List[int] = down_channels[::-1]
    u_conv_num_per_step: int = 2

    ''' Output '''
    output_channel:int = 1


class DownSampleLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, conv_num: int):
        super().__init__()

        self.downsample_layers = nn.Sequential()

        self.downsample_layers.append(nn.Conv2d(in_channel, out_channel, 3, 1, 1))
        self.downsample_layers.append(nn.BatchNorm2d(out_channel))
        self.downsample_layers.append(nn.ReLU(True))

        for _ in range(conv_num-1):
            self.downsample_layers.append(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
            self.downsample_layers.append(nn.BatchNorm2d(out_channel))
            self.downsample_layers.append(nn.ReLU(True))
    
    def forward(self, x):
        return self.downsample_layers(x)


class UpSampleLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, conv_num: int):
        super().__init__()

        # Upsampling and decreasing channels
        self.up_decrease_channels = nn.Sequential()

        self.up_decrease_channels.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.up_decrease_channels.append(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
        
        # Conv2d
        self.conv2d_layers = nn.Sequential()

        self.conv2d_layers.append(nn.Conv2d(out_channel*2, out_channel, 3, 1, 1))
        self.conv2d_layers.append(nn.BatchNorm2d(out_channel))
        self.conv2d_layers.append(nn.ReLU(True))

        for _ in range(conv_num-1):
            self.conv2d_layers.append(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
            self.conv2d_layers.append(nn.BatchNorm2d(out_channel))
            self.conv2d_layers.append(nn.ReLU(True))
    
    def forward(self, x, downsample_feature):
        x = self.up_decrease_channels(x)

        return self.conv2d_layers(torch.concat([x, downsample_feature], 1))


class OutputLayer(nn.Module):
    def __init__(self, in_channel: int, out_channle: int):
        super().__init__()

        self.out_conv2d_1 = self._create_out_conv2d_layer(in_channel, in_channel//2)
        self.out_conv2d_2 = self._create_out_conv2d_layer(in_channel//2, in_channel//4)
        self.out_conv2d_3 = self._create_out_conv2d_layer(in_channel//4, in_channel//8)

        self.outputs = self._create_output_layer(in_channel//8, out_channle)
    
    def _create_out_conv2d_layer(self, in_channel: int, out_channle: int) -> nn.Sequential:
        layers = []

        layers.append(nn.Conv2d(in_channel, out_channle, 1, 1, 0))
        layers.append(nn.BatchNorm2d(out_channle))
        layers.append(nn.ReLU(True))

        return nn.Sequential(*layers)

    def _create_output_layer(self, in_channel: int, out_channle: int) -> nn.Sequential:
        layers = []

        layers.append(nn.Conv2d(in_channel, out_channle, 1, 1, 0))
        layers.append(nn.BatchNorm2d(out_channle))
        layers.append(nn.Sigmoid())
        # layers.append(nn.ReLU(True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.out_conv2d_1(x)
        x = self.out_conv2d_2(x)
        x = self.out_conv2d_3(x)

        x = self.outputs(x)

        return x


class Unet(nn.Module):
    def __init__(self, paras: Type[UnetParas]):
        super().__init__()
        ''' Downsampling '''
        self.downsampling = self._create_downsample_layer(DownSampleLayer, 
                                                          paras.in_data_channel, paras.down_channels, paras.d_conv_num_per_step)
        self.maxpool2d = nn.ModuleList()
        for _ in range(len(paras.down_channels)-1):
            self.maxpool2d.append(nn.MaxPool2d(2))

        ''' Upsampling '''
        self.upsampling = self._create_upsample_layer(UpSampleLayer, paras.up_channels, paras.u_conv_num_per_step)

        '''Output'''
        self.output = OutputLayer(paras.up_channels[-1], paras.output_channel)

    def _create_downsample_layer(self, down_sample_layer: Type[DownSampleLayer], 
                                 in_data_channel: int, down_channels: List[int], conv_num: int) -> nn.ModuleList:
        layers = nn.ModuleList()

        channels = [in_data_channel] + down_channels

        for idx in range(len(down_channels)):
            layers.append(down_sample_layer(channels[idx], channels[idx+1], conv_num))
        
        return layers


    def _create_upsample_layer(self, up_sample_layer: Type[UpSampleLayer],
                               up_channels: List[int], conv_num: int) -> nn.ModuleList:
        layers = nn.ModuleList()

        for idx in range(len(up_channels)-1):
            layers.append(up_sample_layer(up_channels[idx], up_channels[idx+1], conv_num))

        return layers


    def _forward(self, x):
        ''' Downsampling '''
        down_sample_features = []

        for idx in range(len(self.downsampling)-1):
            x = self.downsampling[idx](x)

            down_sample_features.append(x)

            x = self.maxpool2d[idx](x)
        
        x = self.downsampling[-1](x)

        down_sample_features = down_sample_features[::-1]


        ''' Upsampling '''
        for idx in range(len(self.upsampling)):
            x = self.upsampling[idx](x, down_sample_features[idx])
        

        ''' Output '''
        x = self.output(x)
        
        return x
    
    def forward(self, x):
        return self._forward(x)
    

if '__main__' == __name__:
    unetParas = UnetParas()

    unet = Unet(unetParas)

    summary(unet, (unetParas.in_data_channel, unetParas.in_data_height, unetParas.in_data_width), 4, 'cpu')
