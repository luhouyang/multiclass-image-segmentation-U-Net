import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class UNET(nn.Module):

    def __init__(self, in_channels=3, classes=1):
        super(UNET, self).__init__()

        self.layers = [in_channels, 64, 128, 256, 512, 1024]

        # feature extraction
        """
        in:     [in_channels, 64 , 128, 256, 512]
        out:    [64         , 128, 256, 512, 1024]
        """
        self.double_conv_downs = nn.ModuleList([
            self.__double_conv2d(layer, layer_n)
            for layer, layer_n in zip(self.layers[:-1], self.layers[1:])
        ])

        # transpose (scale down) output of down Conv2d layers to match up Conv2d
        """
        in:     [1024, 512, 256, 128]
        out:    [512 , 256, 128, 64]
        """
        self.up_trans = nn.ModuleList([
            nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
            for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1]
                                      [1:-1])
        ])

        # decoder
        """
        in:     [1024, 512, 256, 128]
        out:    [512 , 256, 128, 64]
        """
        self.double_conv_ups = nn.ModuleList([
            self.__double_conv2d(layer, layer // 2)
            for layer in self.layers[::-1][:-2]
        ])

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def __double_conv2d(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        return conv

    def forward(self, x):
        """
        FORMAT: (in, out) | (out)

        down (in_channels, 64)
        down (64, 128)
        down (128, 256)
        down (256, 512)
        down (512, 1024)

        concat_layers = [512, 256, 128, 64, in_channels]

        trans (1024, 512) + concat_layers[0] (512)
        up (512 + 512 = 1024, 512)
        trans (512, 256) + concat_layers[1] (256)
        up (256 + 256 = 512, 256)
        trans (256, 128) + concat_layers[2] (128)
        up (128 + 128 = 256, 128)
        trans (128, 64) + concat_layers[3] (64)
        up (64 + 64 = 128, 64)
        """

        # down layers, feature extraction
        concat_layers = []

        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)

        concat_layers = concat_layers[::-1]

        # up layers, image segmentation
        for up_tran, double_conv_up, concat_layer in zip(
                self.up_trans, self.double_conv_ups, concat_layers):
            x = up_tran(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])

            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        x = self.final_conv(x)

        return x
