import torch.nn as nn

def init_weight(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') == 1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') == 1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    '''
    定义Generator生成器类，神经网络架构为全连接层+转置卷积层
    '''
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                # output_size: (64 - 1) * stride(2) - 2 * padding(2) + kernel_size(5)-1 + output_padding(1) + 1 = 128; 64 -> 128
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.Layer1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4),  # 100 -> 深度dim*8, 大小4*4的张量
            nn.BatchNorm2d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.Layer2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.apply(init_weight)
    def forward(self, x):
        y = self.Layer1(x)                  # 当前y为torch.Size([N, dim*8*4*4])
        y = y.view(y.size(0), -1, 4, 4)     # 重塑y张量，为torch.Size([N, dim*8, 4, 4])
        y = self.Layer2_5(y)                # 将重塑完的四维张量y输入到转置卷积层中，生成fake_img
        return y
class Discriminator(nn.Module):
    '''
    定义Discriminator类，本质上为基于CNN的图像分类器
    '''
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        def conv_bn_lreru(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU()
            )
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),
            nn.LeakyReLU(),
            conv_bn_lreru(dim, dim*2),
            conv_bn_lreru(dim * 2, dim * 4),
            conv_bn_lreru(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid(),
        )
        self.apply(init_weight)
    def forward(self, x):
        y = self.layers(x)
        y = y.view(-1)
        return y
