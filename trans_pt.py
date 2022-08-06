import os
import numpy as np
import torch
from util import save_image, load_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
import onnx
import platform

if platform.system() != "Darwin":
    import onnxruntime
import netron
from torch.utils.mobile_optimizer import optimize_for_mobile
from PIL import Image
import io
import time


def save_jit():
    device = "cpu"
    # root = "/kaggle/input/zhoudualstylegan/DualStyleGAN/"
    root = ""
    pt_path = root + "checkpoint/head2-copy/generator-001500.pt"
    # generator = DualStyleGAN()
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    ckpt = torch.load(pt_path, map_location=lambda storage, loc: storage)
    # netron.start("checkpoint/head2-copy/generator-001500.pt")

    # "g_ema"是训练结果保存进去的约定值
    generator.load_state_dict(ckpt["g_ema"])
    generator.eval()
    generator = generator.to(device)
    exstyles = np.load(root + "checkpoint/head2-copy/refined_exstyle_code.npy", allow_pickle=True).item()

    model_path = os.path.join(root + 'checkpoint', 'encoder.pt')
    ckpt = torch.load(model_path, map_location=device)
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)
    # torch.onnx.symbolic_registry.UnsupportedOperatorError: Exporting the operator prim::layout to ONNX opset version 11 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub.
    with torch.no_grad():
        img = load_image('./data/content/unsplash-rDEOVtE7vOs.jpg').to(device)
        # img = F.adaptive_avg_pool2d(img, 256)
        # instyle = encoder(img)
        # instyle = encoder(F.adaptive_avg_pool2d(img, 256), randomize_noise=False, return_latents=True,
        #                      z_plus_latent=True, return_z_plus_latent=True, resize=False)
        instyle = encoder(img)
        # instyle = encoder(F.adaptive_avg_pool2d(img, 256))
        "head2-copy-mobile_model_encoder.ptl"
        # print(instyle.shape)
        # traced_script_module_encoder = torch.jit.trace(encoder, img, check_trace=False)
        # traced_script_module_encoder.save("head2-copy_model_encoder.jit")
        # traced_script_module_optimized_encoder = optimize_for_mobile(traced_script_module_encoder, backend='Vulkan')
        # traced_script_module_optimized_encoder = optimize_for_mobile(traced_script_module_encoder)
        # traced_script_module_optimized_encoder._save_for_lite_interpreter("head2-copy-mobile_model_encoder.ptl")
        # instyle = torch.jit.load("head2-copy-mobile_model_encoder.ptl")(img)
        stylename = list(exstyles.keys())[5]
        # print("stylename", stylename)
        # print("exstyles[stylename]", exstyles[stylename])
        latent = torch.tensor(exstyles[stylename]).to(device)
        # # # print("latent", latent)
        # print("traced_script_module", "为什么这里什么也没有1？latent ", latent)
        # exstyles = generator.generator.style(
        #     latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
        #     latent.shape)
        # torch.save(latent, './head2-copy_latent.pt')
        # # torch.save(exstyles, './head2-copy_exstyles.pt')
        # # extrinsic styte code
        # print("traced_script_module", "为什么这里什么也没有2？instyle", instyle)
        traced_script_module = torch.jit.trace(generator, (instyle, latent), check_trace=False)
        # traced_script_module = torch.jit.trace(generator, (instyle, exstyles), check_trace=True)
        print("traced_script_module", traced_script_module)
        traced_script_module.save("head2-copy_model.jit")

        # traced_script_module_optimized = optimize_for_mobile(traced_script_module, backend='Vulkan')
        # traced_script_module_optimized._save_for_lite_interpreter("head2-copy-mobile_model.ptl")

        # 报错如下
        # Could not export Python function call 'FusedLeakyReLUFunction'. Remove calls to Python functions before export.
        # Did you forget to add @script or @script_method annotation? If this is a nn.ModuleList, add it to __constants__:


def load_jit():
    model = torch.jit.load("head2-copy_model.jit")
    print(model)
    model.eval()


def save_latent():
    I = load_image("data/head2/images/train/Image_96.JPG").to("cpu")
    model = torch.jit.load("checkpoint/faces_w_encoder.jit")
    print("IIIII", F.adaptive_avg_pool2d(I, 256).shape)
    latent = model(F.adaptive_avg_pool2d(I, 256))
    torch.save(latent, "head3_exstyles_latent.pth")


def trans_tensor(latent: torch.Tensor, path: str):
    dim0, dim1, dim2 = latent.shape
    print(latent.shape)

    filename = open(path, 'w')
    data = []
    # 遍历张量
    for i in range(dim0):
        for j in range(dim1):
            for a in range(dim2):
                element = latent.data[i][j][a]
                data.append(element)
                filename.write(str(element.item()))
                filename.write('\n')

                # print(element.item())

    filename.close()


def save_jit1():
    device = "cpu"
    root = "/home/zhou/Downloads/models/head2-copy"
    pt_path = os.path.join(root, "generator.pt")
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    ckpt = torch.load(pt_path, map_location=lambda storage, loc: storage)
#
    # "g_ema"是训练结果保存进去的约定值
    generator.load_state_dict(ckpt["g_ema"])
    generator.eval()
    generator = generator.to(device)
    if os.path.exists(os.path.join(root, "refined_exstyle_code.npy")):
        exstyles = np.load(os.path.join(root, "refined_exstyle_code.npy"), allow_pickle=True).item()
    else:
        exstyles = np.load(os.path.join(root, "exstyle_code.npy"), allow_pickle=True).item()

    with torch.no_grad():
        img = load_image('./data/content/unsplash-rDEOVtE7vOs.jpg').to(device)
        encoder = torch.jit.load("/home/zhou/Downloads/models/faces_w_encoder.jit")
        instyle = encoder(F.adaptive_avg_pool2d(img, 256))
        print("instyle", instyle.shape)
        stylename = list(exstyles.keys())[1]
        latent = torch.tensor(exstyles[stylename]).to(device)
        trans_tensor(latent, os.path.join(root, "latent.pt"))

        traced_script_module = torch.jit.trace(generator, (instyle, latent), check_trace=False)
        traced_script_module_optimized = optimize_for_mobile(traced_script_module, backend='cpu')
        traced_script_module_optimized._save_for_lite_interpreter(os.path.join(root, "mobile_model.ptl"))


if __name__ == "__main__":
    # save_jit()
    save_jit1()
    # load_jit()
    # save_latent()
