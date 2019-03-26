# WidthParam = 1.0
# Remove the last two dw
# So the output is 14x14x512, same with VGG

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdeNet import HDENet 
from collections import namedtuple, OrderedDict

# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [                                          # layer, factor 
    Conv(kernel=[3, 3], stride=2, depth=32),            # 0,        2
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),    # 1,
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),   # 2,        4
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),   # 3,   
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),   # 4,        8
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),   # 5,    
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),   # 6,        16
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),   # 7,    
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),   # 8,        32
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),   # 9,    
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),   # 10,       64
    DepthSepConv(kernel=[3, 3], stride=1, depth=512)    # 11,   
    # DepthSepConv(kernel=[3, 3], stride=2, depth=1024),# 12
    # DepthSepConv(kernel=[3, 3], stride=1, depth=1024) # 13
]


def mobilenet_v1_base(final_endpoint='Conv2d_11_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=_CONV_DEFS):
    """Mobilenet v1.
    Constructs a Mobilenet v1 network from inputs to the given final endpoint.

    Args:
        inputs: a tensor of shape [batch_size, height, width, channels].
        final_endpoint: specifies the endpoint to construct the network up to. It
            can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
            'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5_pointwise',
            'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
            'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
            'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
        min_depth: Minimum depth value (number of channels) for all convolution ops.
            Enforced when depth_multiplier < 1, and not an active constraint when
            depth_multiplier >= 1.
        depth_multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops. The value must be greater than zero. Typical
            usage will be to set this value in (0, 1) to reduce the number of
            parameters or computation cost of the model.
        conv_defs: A list of ConvDef namedtuples specifying the net architecture.

    Returns:
        tensor_out: output tensor corresponding to the final_endpoint.
        end_points: a set of activations for external use, for example summaries or
                                losses.

    Raises:
        ValueError: if final_endpoint is not set to one of the predefined values,
                                or depth_multiplier <= 0
    """
    def conv_bn(in_channels, out_channels, kernel_size=3, stride=1):
        ### standard convolution
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU6(inplace=True)
        )

    def conv_dw(in_channels, kernel_size=3, stride=1, dilation=1):
        ### depthwise convolution
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 1,
                      groups=in_channels, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels, eps=0.001),
            nn.ReLU6(inplace=True)
        )

    def conv_pw(in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
        ### pointwise convolution
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU6(inplace=True)
        )

    def depth(d):
        return max(int(d * depth_multiplier), min_depth)

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    end_points = OrderedDict()

    # dilation rate parameter
    layer_rate = 1

    in_channels = 3
    for i, conv_def in enumerate(conv_defs):
        end_point_base = 'Conv2d_{}'.format(i)

        layer_stride = conv_def.stride
        out_channels = depth(conv_def.depth)

        if isinstance(conv_def, Conv):
            end_point = end_point_base
            end_points[end_point] = conv_bn(in_channels, out_channels, conv_def.kernel,
                                            stride=conv_def.stride)
            if end_point == final_endpoint:
                return nn.Sequential(end_points)

        elif isinstance(conv_def, DepthSepConv):
            end_points[end_point_base] = nn.Sequential(OrderedDict([
                ('depthwise', conv_dw(in_channels, conv_def.kernel,
                                      stride=layer_stride, dilation=layer_rate)),
                ('pointwise', conv_pw(in_channels, out_channels, 1, stride=1))]))

            if end_point_base + '_pointwise' == final_endpoint:
                return nn.Sequential(end_points)

        else:
            raise ValueError('Unknown convolution type %s for layer %d'
                             % (conv_def.ltype, i))
        in_channels = out_channels
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


class MobileNet_v1(HDENet):

    def __init__(self,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=_CONV_DEFS,
                 device=None):
        """Mobilenet v1 model for features extraction
        Args:
            min_depth: Minimum depth value (number of channels) for all convolution ops.
                Enforced when depth_multiplier < 1, and not an active constraint when
                depth_multiplier >= 1.
            depth_multiplier: Float multiplier for the depth (number of channels)
                for all convolution ops. The value must be greater than zero. Typical
                usage will be to set this value in (0, 1) to reduce the number of
                parameters or computation cost of the model.
            conv_defs: A list of ConvDef namedtuples specifying the net architecture.
        """
        HDENet.__init__(self, device)

        self.features = mobilenet_v1_base(min_depth=min_depth,
                                          depth_multiplier=depth_multiplier,
                                          conv_defs=conv_defs)

    def forward(self, x):
        x = self.features(x)
        return x
