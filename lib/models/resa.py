import torch.nn as nn
import torch
import torch.nn.functional as F
from  .cc_attention import CrissCrossAttention


class RESA(nn.Module):
    def __init__(self, fea_stride, img_height, img_width):
        super(RESA, self).__init__()
        self.iter = 5
        chan = 512
        fea_stride = fea_stride
        self.height = img_height // fea_stride + 1   # img_height 360
        self.width = img_width // fea_stride       # img_width 640
        self.alpha = 2.0
        conv_stride = 9
        # self.cc_attention = CrissCrossAttention(512)

        for i in range(self.iter):
            conv_vert1 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride // 2), groups=1, bias=False)
            conv_vert2 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride // 2), groups=1, bias=False)

            setattr(self, 'conv_d' + str(i), conv_vert1)
            setattr(self, 'conv_u' + str(i), conv_vert2)

            conv_hori1 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride // 2, 0), groups=1, bias=False)
            conv_hori2 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride // 2, 0), groups=1, bias=False)

            setattr(self, 'conv_r' + str(i), conv_hori1)
            setattr(self, 'conv_l' + str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //
                     2 ** (self.iter - i)) % self.height
            setattr(self, 'idx_d' + str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //
                     2 ** (self.iter - i)) % self.height
            setattr(self, 'idx_u' + str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_r' + str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_l' + str(i), idx_l)

    def forward(self, x):
        x = x.clone()

        for direction in ['d', 'u']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                # print(x.shape)
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))
                

        for direction in ['r', 'l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                # print(x.shape)
                x.add_(self.alpha * F.relu(conv(x[..., idx])))
        return x
