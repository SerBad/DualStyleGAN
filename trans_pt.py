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
import onnxruntime
import netron
from torch.utils.mobile_optimizer import optimize_for_mobile
from PIL import Image
import io


def save_jit():
    device = "cuda"
    root = "/kaggle/input/zhoudualstylegan/DualStyleGAN/"
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
    ckpt = torch.load(model_path, map_location='cuda')
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
        _, instyle = encoder(F.adaptive_avg_pool2d(img, 256), randomize_noise=False, return_latents=True,
                             z_plus_latent=True, return_z_plus_latent=True, resize=False)
        "head2-copy-mobile_model_encoder.ptl"

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
        # print("latent", latent)

        # extrinsic styte code
        print("traced_script_module", "为什么这里什么也没有？", instyle)
        traced_script_module = torch.jit.trace(generator, (instyle, latent), check_inputs=(instyle, latent),
                                               check_trace=False)
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


if __name__ == "__main__":
    save_jit()
    load_jit()
