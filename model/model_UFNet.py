import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(self.relu1(x)))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(self.relu1(x)))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SepConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, bias=True, padding_mode='zeros', depth_multiplier=1):
        super(SepConv2d, self).__init__()

        intermediate_channels = in_channels * depth_multiplier

        self.spatialConv = nn.Conv2d(in_channels, intermediate_channels, kernel_size, stride,
                                     padding, dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)

        self.pointConv = nn.Conv2d(intermediate_channels, out_channels,
                                   kernel_size=1, stride=1, padding=0, dilation=1, bias=bias, padding_mode=padding_mode)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.spatialConv(x)
        x = self.relu(x)
        x = self.pointConv(x)

        return x


conv_dict = {
    'CONV2D': nn.Conv2d,
    'SEPARABLE': SepConv2d
}


class _AtrousModule(nn.Module):
    def __init__(self, conv_type, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_AtrousModule, self).__init__()
        self.conv = conv_dict[conv_type]
        self.atrous_conv = self.conv(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False, padding_mode='zeros')

        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class ASP(nn.Module):
    def __init__(self, conv_type, inplanes, planes):
        super(ASP, self).__init__()

         # WASP
        dilations = [1, 6, 12, 18]
        # dilations = [1, 12, 24, 36]

        # convs = conv_dict[conv_type]


        BatchNorm = nn.BatchNorm2d

        self.aspp1 = _AtrousModule(conv_type, inplanes, planes, 1, padding=0, dilation=dilations[0],
                                   BatchNorm=BatchNorm)
        self.aspp2 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations[1], dilation=dilations[1],
                                   BatchNorm=BatchNorm)
        self.aspp3 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations[2], dilation=dilations[2],
                                   BatchNorm=BatchNorm)
        self.aspp4 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations[3], dilation=dilations[3],
                                   BatchNorm=BatchNorm)

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(planes, planes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(planes),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(5 * planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(2 * planes, planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # adopt [1x1, 48] for channel reduction.

        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()


    def forward(self, x):
        residual =  self.ca(x) * x
        low_level = self.sa(residual) * x# 通道注意力机制，并通过元素乘法与原始输入融合
        # ASP模块处理
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        asp_out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        asp_out = self.conv1(asp_out)
        asp_out = self.bn1(asp_out)
        x = self.relu(asp_out)

        fuse = torch.cat((low_level, x), dim=1)
        fuse_out = self.conv2(fuse)
        fuse_out = self.bn2(fuse_out)
        x = self.relu(fuse_out)
        # 融合注意力机制输出和ASP输出
        # x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class UFNet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter, deep_supervision=False):
        super(DNANet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.down_4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.down_8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1] + nb_filter[2] + nb_filter[3] + nb_filter[4],
                                        nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0] + nb_filter[3] + nb_filter[4],
                                        nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[0] + nb_filter[4] + nb_filter[2] + nb_filter[3] + nb_filter[1],
                                        nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2] + nb_filter[0] + nb_filter[1],
                                        nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1] + nb_filter[2] + nb_filter[3], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2] + nb_filter[0] + nb_filter[3], nb_filter[1],
                                        num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3] + nb_filter[1] + nb_filter[0], nb_filter[2],
                                        num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1] + nb_filter[2], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.ASP = ASP('CONV2D', nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0), self.up_4(x2_0), self.up_8(x3_0), self.up_16(x4_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1), self.up_4(x3_0), self.up_8(x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down_4(x0_1), self.down(x1_1), self.up_4(x4_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1), self.down_4(x1_1), self.down_8(x0_1)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1), self.up_4(x2_1), self.up_8(x3_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.up_4(x3_1), self.down(x0_2)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2), self.down_4(x0_2)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2), self.up_4(x2_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Backbone = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                                 self.up_4(self.conv0_2_1x1(x2_2)), self.up(self.conv0_1_1x1(x1_3)), x0_4], 1))

        Final_x0_4 = self.ASP(Backbone)


        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(Final_x0_4)
            return output


# class DNANet(nn.Module):
#     def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=False):
#         super(DNANet, self).__init__()
#         self.relu = nn.ReLU(inplace = True)
#         self.deep_supervision = deep_supervision
#         self.pool  = nn.MaxPool2d(2, 2)
#         self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
#         self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
#
#         self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
#         self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
#         self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)
#
#         self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
#         self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
#         self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
#         self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
#         self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])
#
#         self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
#         self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
#         self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
#         self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])
#
#         self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
#         self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
#         self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])
#
#         self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
#         self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
#
#         self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])
#
#         self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])
#
#         self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
#         self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
#         self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
#         self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
#         self.ASP = ASP('CONV2D', nb_filter[0], nb_filter[0])
#
#         if self.deep_supervision:
#             self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
#             self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
#             self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
#             self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
#         else:
#             self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
#
#     def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
#         layers = []
#         layers.append(block(input_channels, output_channels))
#         for i in range(num_blocks-1):
#             layers.append(block(output_channels, output_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, input):
#         x0_0 = self.conv0_0(input)
#         x1_0 = self.conv1_0(self.pool(x0_0))
#         x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
#
#         x2_0 = self.conv2_0(self.pool(x1_0))
#         x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
#         x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
#
#         x3_0 = self.conv3_0(self.pool(x2_0))
#         x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))
#         x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))
#         x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
#
#         x4_0 = self.conv4_0(self.pool(x3_0))
#         x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down(x2_1)], 1))
#         x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
#
#         Final_x0_4 = self.conv0_4_final((torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
#                        self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1)))
#         Final_x0_4 = self.ASP(Final_x0_4)
#
#         if self.deep_supervision:
#             output1 = self.final1(x0_1)
#             output2 = self.final2(x0_2)
#             output3 = self.final3(x0_3)
#             output4 = self.final4(Final_x0_4)
#             return [output1, output2, output3, output4]
#         else:
#             output = self.final(Final_x0_4)
#             return output


