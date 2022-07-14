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

if __name__ == "__main__":
    device = "cuda"
    root = "/kaggle/input/zhoudualstylegan/DualStyleGAN/"
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    ckpt = torch.load(root+"checkpoint/head2-copy/generator-001500.pt", map_location=lambda storage, loc: storage)
    # netron.start("checkpoint/head2-copy/generator-001500.pt")

    # "g_ema"是训练结果保存进去的约定值
    generator.load_state_dict(ckpt["g_ema"])
    generator.eval()
    generator = generator.to(device)
    exstyles = np.load(root+"checkpoint/head2-copy/refined_exstyle_code.npy", allow_pickle='TRUE').item()

    model_path = os.path.join(root+'checkpoint', 'encoder.pt')
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
        I = load_image('./data/content/unsplash-rDEOVtE7vOs.jpg').to(device)
        # instyle = encoder(I)
        _, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True,
                             z_plus_latent=True, return_z_plus_latent=True, resize=False)

        stylename = list(exstyles.keys())[5]
        # print("stylename", stylename)
        # print("exstyles[stylename]", exstyles[stylename])
        latent = torch.tensor(exstyles[stylename]).to(device)
        # print("latent", latent)

        # extrinsic styte code
        # exstyle = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
        #     latent.shape)

        # return_latents = False,
        # return_feat = False,
        # inject_index = None,
        # truncation = 0.75,
        # truncation_latent = 0,
        # input_is_latent = False,
        # noise = None,
        # randomize_noise = True,
        # z_plus_latent = True,  # intrinsic style code is z+ or z
        # use_res = True,  # whether to use the extrinsic style path
        # fuse_index = 18,  # layers > fuse_index do not use the extrinsic style path
        # interp_weights = [0.75] * 7 + [1] * 11,  # weight vector for style combination of two paths
        #
        # # weight vector for style combination of two paths
        # [instyle, exstyle, False, False, None, 0.75, 0, False, None, True, True, True, 18,
        #  [0.75] * 7 + [1] * 11]
        # transforms.ToPILImage()(img_rec).show()

        print("instyle", instyle)
        input_names = ["input"]
        output_names = ["output"]
        path = "./head2-copy2.onnx"
        torch.onnx.export(generator,
                          (instyle, latent),
                          path,
                          verbose=True,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          keep_initializers_as_inputs=True )

        print(onnx.checker.check_model(onnx.load(path)))

        session = onnxruntime.InferenceSession(path)
        print("session.get_inputs()", session.get_inputs())
        for o in session.get_inputs():
            print(o)
        for o in session.get_outputs():
            print("session.get_outputs()", o)

        # 可视化模型
        # https://netron.app/
        #
        # torch.onnx.export(generator,
        #                   ([instyle], exstyle, False, False, None, 0.75, 0, False, None, True, True, True, 18,
        #                    [0.75] * 7 + [1] * 11),
        #                   path,
        #                   export_params=True,
        #                   opset_version=16,
        #                   do_constant_folding=True,
        #                   input_names=['input'],
        #                   output_names=['output'])
