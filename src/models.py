"""Все архитектуры проекта"""


import torch
import torch.nn as nn


# ============================================================
# Простая CNN
# ============================================================
def create_simple_cnn() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),
        nn.Linear(8 * 8 * 64, 512),
        nn.Linear(512, 2),
    )


# ============================================================
# ResNet18 
# ============================================================
class BasicBlock_1(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.first_conv2d = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.first_norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.second_conv2d = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.second_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.act(self.first_norm(self.first_conv2d(x)))
        y = self.second_norm(self.second_conv2d(y))
        return self.act(x + y)


class BasicBlock_2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        out = in_channels * 2
        self.first_conv2d = nn.Conv2d(in_channels, out, 3, 2, 1, bias=False)
        self.first_norm = nn.BatchNorm2d(out)
        self.second_conv2d = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.second_norm = nn.BatchNorm2d(out)
        self.act = nn.ReLU()
        self.skip_connection_conv2d = nn.Conv2d(in_channels, out, 1, 2, 0, bias=False)
        self.skip_connection_norm = nn.BatchNorm2d(out)

    def forward(self, x):
        y = self.act(self.first_norm(self.first_conv2d(x)))
        y = self.second_norm(self.second_conv2d(y))
        x = self.skip_connection_norm(self.skip_connection_conv2d(x))
        return self.act(x + y)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv2d = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.first_norm = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.firs_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # опечатка сохранена!
        self.layer1 = nn.Sequential(BasicBlock_1(64), BasicBlock_1(64))
        self.layer2 = nn.Sequential(BasicBlock_2(64), BasicBlock_1(128))
        self.layer3 = nn.Sequential(BasicBlock_2(128), BasicBlock_1(256))
        self.layer4 = nn.Sequential(BasicBlock_2(256), BasicBlock_1(512))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 2, bias=True)

    def forward(self, x):
        x = self.act(self.first_norm(self.first_conv2d(x)))
        x = self.firs_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.linear(x)


# ============================================================
# ResNet50 
# ============================================================
class BottleneckBlock_1(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.norm_2 = nn.BatchNorm2d(in_channels)
        self.conv2d_3 = nn.Conv2d(in_channels, in_channels * 4, 1, 1, 0, bias=False)
        self.norm_3 = nn.BatchNorm2d(in_channels * 4)
        self.conv2d_skip = nn.Conv2d(in_channels, in_channels * 4, 1, 1, 0, bias=False)
        self.norm_skip = nn.BatchNorm2d(in_channels * 4)

    def forward(self, x):
        y = self.act(self.norm(self.conv2d_1(x)))
        y = self.act(self.norm_2(self.conv2d_2(y)))
        y = self.norm_3(self.conv2d_3(y))
        x = self.norm_skip(self.conv2d_skip(x))
        return self.act(x + y)


class BottleneckBlock_2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        out = in_channels // 4
        self.conv2d_1 = nn.Conv2d(in_channels, out, 1, 1, 0, bias=False)
        self.norm = nn.BatchNorm2d(out)
        self.act = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.norm_2 = nn.BatchNorm2d(out)
        self.conv2d_3 = nn.Conv2d(out, in_channels, 1, 1, 0, bias=False)
        self.norm_3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.act(self.norm(self.conv2d_1(x)))
        y = self.act(self.norm_2(self.conv2d_2(y)))
        y = self.norm_3(self.conv2d_3(y))
        return self.act(x + y)


class BottleneckBlock_3(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False)
        self.norm = nn.BatchNorm2d(in_channels // 2)
        self.act = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, 2, 1, bias=False)
        self.norm_2 = nn.BatchNorm2d(in_channels // 2)
        self.conv2d_3 = nn.Conv2d(in_channels // 2, in_channels * 2, 1, 1, 0, bias=False)
        self.norm_3 = nn.BatchNorm2d(in_channels * 2)
        self.conv2d_skip = nn.Conv2d(in_channels, in_channels * 2, 1, 2, 0, bias=False)
        self.norm_skip = nn.BatchNorm2d(in_channels * 2)

    def forward(self, x):
        y = self.act(self.norm(self.conv2d_1(x)))
        y = self.act(self.norm_2(self.conv2d_2(y)))
        y = self.norm_3(self.conv2d_3(y))
        x = self.norm_skip(self.conv2d_skip(x))
        return self.act(x + y)


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.norm = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BottleneckBlock_1(64),
            BottleneckBlock_2(64 * 4),
            BottleneckBlock_2(64 * 4),
        )
        self.layer2 = nn.Sequential(
            BottleneckBlock_3(64 * 4),
            BottleneckBlock_2(128 * 4),
            BottleneckBlock_2(128 * 4),
            BottleneckBlock_2(128 * 4),
        )
        self.layer3 = nn.Sequential(
            BottleneckBlock_3(128 * 4),
            BottleneckBlock_2(256 * 4),
            BottleneckBlock_2(256 * 4),
            BottleneckBlock_2(256 * 4),
            BottleneckBlock_2(256 * 4),
            BottleneckBlock_2(256 * 4),
        )
        self.layer4 = nn.Sequential(
            BottleneckBlock_3(256 * 4),
            BottleneckBlock_2(512 * 4),
            BottleneckBlock_2(512 * 4),
        )
        self.pool_2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4, 2)

    def forward(self, x):
        x = self.act(self.norm(self.conv2d_1(x)))
        x = self.pool_1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        return self.fc(x)


# ============================================================
# Inception V1
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, activation="gelu"):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x):
        return self.relu(self.batchnorm2d(self.conv2d(x)))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3,
                 reduce_5x5, out_5x5, out_1x1_pooling):
        super().__init__()
        self.branch_1 = ConvBlock(in_channels, out_1x1, 1, 1, 0)
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, reduce_3x3, 1, 1, 0),
            ConvBlock(reduce_3x3, out_3x3, 3, 1, 1),
        )
        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, reduce_5x5, 1, 1, 0),
            ConvBlock(reduce_5x5, out_5x5, 5, 1, 2),
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1_pooling, 1, 1, 0),
        )

    def forward(self, x):
        return torch.cat([
            self.branch_1(x), self.branch_2(x),
            self.branch_3(x), self.branch_4(x)
        ], dim=1)


class InceptionV1(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0),
            ConvBlock(64, 192, 3, 1, 1),
        )
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_3_a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3_b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4_a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4_b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4_c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4_d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4_e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5_a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5_b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.maxpool_1(self.conv_1(x))
        x = self.maxpool_2(self.conv_2(x))
        x = self.inception_3_a(x)
        x = self.inception_3_b(x)
        x = self.maxpool_3(x)
        x = self.inception_4_a(x)
        x = self.inception_4_b(x)
        x = self.inception_4_c(x)
        x = self.inception_4_d(x)
        x = self.inception_4_e(x)
        x = self.maxpool_4(x)
        x = self.inception_5_a(x)
        x = self.inception_5_b(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)


# ============================================================
# Inception V3
# ============================================================
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_channels):
        super().__init__()
        self.branch_1 = ConvBlock(in_channels, 64, 1, 1, 0)
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, 48, 1, 1, 0),
            ConvBlock(48, 64, 5, 1, 2),
        )
        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, 64, 1, 1, 0),
            ConvBlock(64, 96, 3, 1, 1),
            ConvBlock(96, 96, 3, 1, 1),
        )
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, pool_channels, 1, 1, 0),
        )

    def forward(self, x):
        return torch.cat([self.branch_1(x), self.branch_2(x),
                          self.branch_3(x), self.branch_4(x)], dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch_1 = ConvBlock(in_channels, 192, 1, 1, 0)
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, 128, 1, 1, 0),
            ConvBlock(128, 128, (1, 7), 1, (0, 3)),
            ConvBlock(128, 192, (7, 1), 1, (3, 0)),
        )
        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, 128, 1, 1, 0),
            ConvBlock(128, 128, (7, 1), 1, (3, 0)),
            ConvBlock(128, 128, (1, 7), 1, (0, 3)),
            ConvBlock(128, 128, (7, 1), 1, (3, 0)),
            ConvBlock(128, 192, (1, 7), 1, (0, 3)),
        )
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, 1, 1, 0),
        )

    def forward(self, x):
        return torch.cat([self.branch_1(x), self.branch_2(x),
                          self.branch_3(x), self.branch_4(x)], dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch_1 = ConvBlock(in_channels, 320, 1, 1, 0)
        self.branch_2_1 = ConvBlock(in_channels, 384, 1, 1, 0)
        self.branch_2_2a = ConvBlock(384, 384, (1, 3), 1, (0, 1))
        self.branch_2_2b = ConvBlock(384, 384, (3, 1), 1, (1, 0))
        self.branch_3_1 = nn.Sequential(
            ConvBlock(in_channels, 448, 1, 1, 0),
            ConvBlock(448, 384, 3, 1, 1),
        )
        self.branch_3_2a = ConvBlock(384, 384, (1, 3), 1, (0, 1))
        self.branch_3_2b = ConvBlock(384, 384, (3, 1), 1, (1, 0))
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, 1, 1, 0),
        )

    def forward(self, x):
        b1 = self.branch_1(x)
        b2 = self.branch_2_1(x)
        b2 = torch.cat([self.branch_2_2a(b2), self.branch_2_2b(b2)], dim=1)
        b3 = self.branch_3_1(x)
        b3 = torch.cat([self.branch_3_2a(b3), self.branch_3_2b(b3)], dim=1)
        b4 = self.branch_4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch_1 = ConvBlock(in_channels, 384, 3, 2, 0)
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, 64, 1, 1, 0),
            ConvBlock(64, 96, 3, 1, 1),
            ConvBlock(96, 96, 3, 2, 0),
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x)], dim=1)


class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch_1 = nn.Sequential(
            ConvBlock(in_channels, 192, 1, 1, 0),
            ConvBlock(192, 320, 3, 2, 0),
        )
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, 192, 1, 1, 0),
            ConvBlock(192, 192, (1, 7), 1, (0, 3)),
            ConvBlock(192, 192, (7, 1), 1, (3, 0)),
            ConvBlock(192, 192, 3, 2, 0),
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x)], dim=1)


class InceptionV3(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_1 = ConvBlock(in_channels, 32, 3, 2, 0)
        self.conv_2 = ConvBlock(32, 32, 3, 1, 0)
        self.conv_3 = ConvBlock(32, 64, 3, 1, 1)
        self.maxpool_1 = nn.MaxPool2d(3, stride=2)
        self.conv_4 = ConvBlock(64, 80, 1, 1, 0)
        self.conv_5 = ConvBlock(80, 192, 3, 1, 0)
        self.maxpool_2 = nn.MaxPool2d(3, stride=2)
        self.inception_3_a = InceptionA(192, 32)
        self.inception_3_b = InceptionA(256, 64)
        self.inception_3_c = InceptionA(288, 64)
        self.reduction_a = ReductionA(288)
        self.inception_4_a = InceptionB(768)
        self.inception_4_b = InceptionB(768)
        self.inception_4_c = InceptionB(768)
        self.inception_4_d = InceptionB(768)
        self.reduction_b = ReductionB(768)
        self.inception_5_a = InceptionC(1280)
        self.inception_5_b = InceptionC(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool_1(self.conv_3(self.conv_2(self.conv_1(x))))
        x = self.maxpool_2(self.conv_5(self.conv_4(x)))
        x = self.inception_3_c(self.inception_3_b(self.inception_3_a(x)))
        x = self.reduction_a(x)
        x = self.inception_4_d(self.inception_4_c(self.inception_4_b(self.inception_4_a(x))))
        x = self.reduction_b(x)
        x = self.inception_5_b(self.inception_5_a(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================
# UNet денойзер
# ============================================================
class UNetForDenoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.max_pool_1 = nn.MaxPool2d(2)
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.max_pool_2 = nn.MaxPool2d(2)
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.max_pool_3 = nn.MaxPool2d(2)
        self.botneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.upconv_3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.upconv_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.upconv_1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.output = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(self.max_pool_1(e1))
        e3 = self.encoder_3(self.max_pool_2(e2))
        b = self.botneck(self.max_pool_3(e3))
        d3 = self.decoder_3(torch.cat([self.upconv_3(b), e3], dim=1))
        d2 = self.decoder_2(torch.cat([self.upconv_2(d3), e2], dim=1))
        d1 = self.decoder_1(torch.cat([self.upconv_1(d2), e1], dim=1))
        return self.output(d1)


# ============================================================
# DenoiseResNet
# ============================================================
class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class DenoiseResNet(nn.Module):
    def __init__(self, cnt_blocks: int):
        super().__init__()
        self.input = nn.Conv2d(3, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[ResNetBlock(64) for _ in range(cnt_blocks)])
        self.output = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)


# ============================================================
# UNet для классификации
# ============================================================
class UNetForClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.max_pool_1 = nn.MaxPool2d(2)
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.max_pool_2 = nn.MaxPool2d(2)
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.max_pool_3 = nn.MaxPool2d(2)
        self.botneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.upconv_3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.upconv_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.upconv_1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(self.max_pool_1(e1))
        e3 = self.encoder_3(self.max_pool_2(e2))
        b = self.botneck(self.max_pool_3(e3))
        d3 = self.decoder_3(torch.cat([self.upconv_3(b), e3], dim=1))
        d2 = self.decoder_2(torch.cat([self.upconv_2(d3), e2], dim=1))
        d1 = self.decoder_1(torch.cat([self.upconv_1(d2), e1], dim=1))
        pooled = self.pool(d1)
        return self.output(pooled.view(pooled.size(0), -1))