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


class TestOptions:
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Exemplar-Based Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/content/unsplash-rDEOVtE7vOs.jpg',
                                 help="path of the content image")
        self.parser.add_argument("--style", type=str, default='head2-copy', help="target style type")
        self.parser.add_argument("-style_id", type=int, default=5, help="the id of the style image")
        self.parser.add_argument("--truncation", type=float, default=0.75,
                                 help="truncation for intrinsic style code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75] * 7 + [1] * 11,
                                 help="weight of the extrinsic style")
        self.parser.add_argument("--name", type=str, default='simpsons_transfer',
                                 help="filename to save the generated images")
        self.parser.add_argument("--preserve_color", action="store_true",
                                 help="preserve the color of the content image")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator-001500.pt',
                                 help="name of the saved dualstylegan")
        self.parser.add_argument("--output_path", type=str, default='./output-test/', help="path of the output images")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--align_face", action="store_true", default=False,
                                 help="apply face alignment to the content image")
        self.parser.add_argument("--exstyle_name", type=str, default=None, help="name of the extrinsic style codes")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_name is None:
            if os.path.exists(os.path.join(self.opt.model_path, self.opt.style, 'refined_exstyle_code.npy')):
                self.opt.exstyle_name = 'refined_exstyle_code.npy'
            else:
                self.opt.exstyle_name = 'exstyle_code.npy'
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt




if __name__ == "__main__":
    device = "cpu"

    parser = TestOptions()
    args = parser.parse()
    print('*' * 98)

    generator = torch.jit.load("head2-copy_model.jit")
    # exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle=True).item()
    exstyles = torch.load("head2-copy_exstyles.pt")
    print('Load models successfully!')

    # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
    with torch.no_grad():
        viz = []
        I = load_image(args.content).to(device)
        viz += [I]

        # instyle = torch.jit.load("head2-copy_model_encoder.jit")(F.adaptive_avg_pool2d(I, 256))
        instyle = torch.jit.load("head2-copy_model_encoder.jit")(I)

        print(instyle.shape)
        # print('exstyles.keys()', exstyles.keys())
        # stylename = list(exstyles.keys())[args.style_id]
        # latent = torch.tensor(exstyles[stylename]).to(device)

        img_gen = generator(instyle, exstyles)
        img_gen = torch.clamp(img_gen.detach(), -1, 1).to(device)
        viz += [img_gen]

    print('Generate images successfully!')

    save_name = args.name + '_%d_%s' % (args.style_id, os.path.basename(args.content).split('.')[0])
    save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 4, 2).cpu(),
               os.path.join(args.output_path, save_name + '_overview.jpg'))
    save_image(img_gen.cpu(), os.path.join(args.output_path, save_name + '.jpg'))

    print('Save images successfully!')
