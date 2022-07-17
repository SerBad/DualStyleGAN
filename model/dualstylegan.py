import random

import torch
from torch import nn
import torchvision.transforms as T

import numpy as np
from model.stylegan.model import ConvLayer, PixelNorm, EqualLinear, Generator


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, fin, style_dim=512):
        super().__init__()

        self.norm = nn.InstanceNorm2d(fin, affine=False)
        self.style = nn.Linear(style_dim, fin * 2)

        self.style.bias.data[:fin] = 1
        self.style.bias.data[fin:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out


# modulative residual blocks (ModRes)
class AdaResBlock(nn.Module):
    def __init__(self, fin, style_dim=512):
        super().__init__()

        self.conv = ConvLayer(fin, fin, 3)
        self.conv2 = ConvLayer(fin, fin, 3)
        self.norm = AdaptiveInstanceNorm(fin, style_dim)
        self.norm2 = AdaptiveInstanceNorm(fin, style_dim)

        # model initialization
        # the convolution filters are set to values close to 0 to produce negligible residual features
        self.conv[0].weight.data *= 0.01
        self.conv2[0].weight.data *= 0.01

    def forward(self, x, s, w=1):
        skip = x
        if w == 0:
            return skip
        out = self.conv(self.norm(x, s))
        out = self.conv2(self.norm2(out, s))
        out = out * w + skip
        return out


class DualStyleGAN(nn.Module):
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, twoRes=True, res_index=6):
        super().__init__()
        print("初始化？", "style_dim", style_dim, "n_mlp", n_mlp)
        layers = [PixelNorm()]
        for i in range(n_mlp - 6):
            print("执行了几遍？")
            # 和StyleGAN一样的全连接层，但这里只用了2个，不知道为什么
            # EqualLinear 基于linear和conv，通过缩放网络权重，使得每一层的参数的学习率能够保持一致，从而能增强GAN的稳定性，改善图像质量。
            layers.append(EqualLinear(512, 512, lr_mul=0.01, activation="fused_lrelu"))
        print("layers的层数", len(layers))
        # color transform blocks T_c
        # nn.Sequential 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        self.style = nn.Sequential(*layers)
        # StyleGAN2
        self.generator = Generator(size, style_dim, n_mlp, channel_multiplier)
        # The extrinsic style path
        self.res = nn.ModuleList()
        self.res_index = res_index // 2 * 2
        self.res.append(AdaResBlock(self.generator.channels[2 ** 2]))  # for conv1
        for i in range(3, self.generator.log_size + 1):
            out_channel = self.generator.channels[2 ** i]
            if i < 3 + self.res_index // 2:
                # ModRes
                self.res.append(AdaResBlock(out_channel))
                self.res.append(AdaResBlock(out_channel))
            else:
                # structure transform block T_s
                self.res.append(EqualLinear(512, 512))
                # FC layer is initialized with identity matrices, meaning no changes to the input latent code
                self.res[-1].weight.data = torch.eye(512) * 512.0 ** 0.5 + torch.randn(512, 512) * 0.01
                self.res.append(EqualLinear(512, 512))
                self.res[-1].weight.data = torch.eye(512) * 512.0 ** 0.5 + torch.randn(512, 512) * 0.01
        self.res.append(EqualLinear(512, 512))  # for to_rgb7
        self.res[-1].weight.data = torch.eye(512) * 512.0 ** 0.5 + torch.randn(512, 512) * 0.01
        self.size = self.generator.size
        self.style_dim = self.generator.style_dim
        self.log_size = self.generator.log_size
        self.num_layers = self.generator.num_layers
        self.n_latent = self.generator.n_latent
        self.channels = self.generator.channels

    def forward(
            self,
            styles,  # intrinsic style code
            exstyles,  # extrinsic style code
            return_latents=False,
            return_feat=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            z_plus_latent=False,  # intrinsic style code is z+ or z
            use_res=True,  # whether to use the extrinsic style path
            fuse_index=18,  # layers > fuse_index do not use the extrinsic style path
            interp_weights=[1] * 18,  # weight vector for style combination of two paths
    ):
        print("这里会执行到吗？", "input_is_latent", input_is_latent, z_plus_latent)
        # styles = styles
        # exstyles = exstyles
        # return_latents = return_latents.item()
        # return_feat = return_feat.item()
        # inject_index = inject_index
        # truncation = truncation.item()
        # truncation_latent = truncation_latent.item()
        # input_is_latent = input_is_latent.item()
        # noise = noise
        # randomize_noise = randomize_noise.item()
        # z_plus_latent = z_plus_latent.item()
        # use_res = use_res.item()
        # fuse_index = fuse_index.item()
        # interp_weights = interp_weights

        styles = [styles]

        return_latents = False
        return_feat = False
        inject_index = None
        truncation = 0.75
        truncation_latent = 0
        input_is_latent = False
        noise = None
        randomize_noise = True
        z_plus_latent = True  # intrinsic style code is z+ or z
        use_res = True  # whether to use the extrinsic style path
        fuse_index = 18  # layers > fuse_index do not use the extrinsic style path
        interp_weights = [0.75] * 7 + [1] * 11  # weight vector for style combination of two paths

        for s in styles:
            print("输入的值", s.shape)

        if not input_is_latent:
            if not z_plus_latent:
                styles = [self.generator.style(s) for s in styles]
            else:
                styles = [self.generator.style(s.reshape(s.shape[0] * s.shape[1], s.shape[2])).reshape(s.shape) for s in
                          styles]
        print("这里会执行到吗？", "noise", noise, 'randomize_noise', randomize_noise)
        if noise is None:
            if randomize_noise:
                noise = [None] * self.generator.num_layers
            else:
                noise = [
                    getattr(self.generator.noises, f"noise_{i}") for i in range(self.generator.num_layers)
                ]
        print("truncation", truncation)

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t
        print("这里执行到了没？回事因为最后输出的问题吗33？")
        if len(styles) < 2:
            inject_index = self.generator.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.generator.n_latent - 1)

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.generator.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)
            else:
                latent = torch.cat([styles[0][:, 0:inject_index], styles[1][:, inject_index:]], 1)
        print("这里执行到了没？回事因为最后输出的问题吗44？")
        if use_res:
            if exstyles.ndim < 3:
                resstyles = self.style(exstyles).unsqueeze(1).repeat(1, self.generator.n_latent, 1)
                adastyles = exstyles.unsqueeze(1).repeat(1, self.generator.n_latent, 1)
            else:
                nB, nL, nD = exstyles.shape
                resstyles = self.style(exstyles.reshape(nB * nL, nD)).reshape(nB, nL, nD)
                adastyles = exstyles

        out = self.generator.input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=noise[0])
        if use_res and fuse_index > 0:
            out = self.res[0](out, resstyles[:, 0], interp_weights[0])
        print("这里执行到了没？回事因为最后输出的问题吗5t5？")
        skip = self.generator.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2], noise[1::2], noise[2::2],
                self.generator.to_rgbs):
            if use_res and fuse_index >= i and i > self.res_index:
                out = conv1(out, interp_weights[i] * self.res[i](adastyles[:, i]) +
                            (1 - interp_weights[i]) * latent[:, i], noise=noise1)
            else:
                out = conv1(out, latent[:, i], noise=noise1)
            if use_res and fuse_index >= i and i <= self.res_index:
                out = self.res[i](out, resstyles[:, i], interp_weights[i])
            if use_res and fuse_index >= (i + 1) and i > self.res_index:
                out = conv2(out, interp_weights[i + 1] * self.res[i + 1](adastyles[:, i + 1]) +
                            (1 - interp_weights[i + 1]) * latent[:, i + 1], noise=noise2)
            else:
                out = conv2(out, latent[:, i + 1], noise=noise2)
            if use_res and fuse_index >= (i + 1) and i <= self.res_index:
                out = self.res[i + 1](out, resstyles[:, i + 1], interp_weights[i + 1])
            if use_res and fuse_index >= (i + 2) and i >= self.res_index - 1:
                skip = to_rgb(out, interp_weights[i + 2] * self.res[i + 2](adastyles[:, i + 2]) +
                              (1 - interp_weights[i + 2]) * latent[:, i + 2], skip)
            else:
                skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
            print("这里执行到了没？回事因为最后输出的问题吗66？", i)
            if i > self.res_index and return_feat:
                print("这里执行到了没？回事因为最后输出的问题吗11？")
                return out, skip

        image = skip
        print("这里执行到了没？回事因为最后输出的问题吗22？", image)
        # image = torch.clamp(image.detach(), -1, 1)[0].cpu()
        # # print(image.shape)
        # # image = image.to(memory_format=torch.preserve_format)
        # # image = T.ToTensor()(image)
        # print(image.layout)
        # # print(exstyles)
        # # image.show()
        # image = T.ToPILImage()(image)
        # return T.ToTensor()(image)
        return image

        # if return_latents:
        #     return image, latent
        # else:
        #     # image = torch.clamp(image.detach(), -1, 1)[0].cpu()
        #     # print(image.shape)
        #     # # image = image.to(memory_format=torch.preserve_format)
        #     # # image = T.ToTensor()(image)
        #     # print(image.layout)
        #     # # print(exstyles)
        #     # # image.show()
        #     # image = T.ToPILImage()(image)
        #     # return T.ToTensor()(image)
        #     return image, None

    def make_noise(self):
        return self.generator.make_noise()

    def mean_latent(self, n_latent):
        return self.generator.mean_latent(n_latent)

    def get_latent(self, input):
        return self.generator.style(input)
