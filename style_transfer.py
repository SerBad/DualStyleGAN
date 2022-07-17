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


class TestOptions:
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Exemplar-Based Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/content/unsplash-rDEOVtE7vOs.jpg',
                                 help="path of the content image")
        self.parser.add_argument("--style", type=str, default='head2-copy', help="target style type")
        self.parser.add_argument("-style_id", type=int, default=3, help="the id of the style image")
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


def run_alignment(args):
    import dlib
    from model.encoder.align_all_parallel import align_face
    modelname = os.path.join(args.model_path, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname + '.bz2')
        zipfile = bz2.BZ2File(modelname + '.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data)
    predictor = dlib.shape_predictor(modelname)
    aligned_image = align_face(filepath=args.content, predictor=predictor)
    return aligned_image


if __name__ == "__main__":
    device = "cpu"

    parser = TestOptions()
    args = parser.parse()
    print('*' * 98)

    # torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起：
    # transforms.ToTensor()能够把灰度范围从0-255变换到0-1之间
    # transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))，则其作用就是先将输入归一化到(0,1)，再使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    # https://zhuanlan.zhihu.com/p/357075502
    # model.eval()的作用是不启用 Batch Normalization 和 Dropout。
    # model.train()的作用是启用 Batch Normalization 和 Dropout。
    # Batch Normalization原理与实战 https://zhuanlan.zhihu.com/p/34879333
    generator.eval()

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name),
                      map_location=lambda storage, loc: storage)
    # "g_ema"是训练结果保存进去的约定值
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)

    # 'encoder.pt'是Pixel2style2pixel model
    # https://github.com/eladrich/pixel2style2pixel
    # https://zhuanlan.zhihu.com/p/267834502
    # （1）将图像转成隐藏编码；（2）将人脸转正；（3）图像合成（根据草图或者分割结果生成图像）；（4）将低分辨率图像转成高清图像。
    # Pixel2style2pixel是基于StyleGAN的latent space进一步的延申
    model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    opts.device = device
    # encoder = pSp(opts)
    # encoder.eval()
    # encoder.to(device)

    # 来源于destylize.py保存下来的exstyle_code.npy
    # 使用encode，也就是pSp处理之后返回的style code z^+_e的集合
    exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle=True).item()

    print('Load models successfully!')

    # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
    with torch.no_grad():
        viz = []
        # load content image
        if args.align_face:
            I = transform(run_alignment(args)).unsqueeze(dim=0).to(device)
            I = F.adaptive_avg_pool2d(I, 1024)
        else:
            I = load_image(args.content).to(device)
        viz += [I]
        # img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True,
        #                            z_plus_latent=True, return_z_plus_latent=True, resize=False)
        # F.adaptive_avg_pool2d自适应平均池化函数
        # reconstructed content image and its intrinsic style code
        model = torch.jit.load("head2-copy_model_encoder.jit")
        # instyle = model(F.adaptive_avg_pool2d(I, 256))
        instyle = model(I)
        # input_names = ["input"]
        # output_names = ["output"]
        # path = "./head2-copy2_encoder.onnx"
        # torch.onnx.export(model,
        #                   F.adaptive_avg_pool2d(I, 256),
        #                   path,
        #                   verbose=True,
        #                   export_params=True,
        #                   opset_version=11,
        #                   do_constant_folding=True,
        #                   input_names=input_names,
        #                   output_names=output_names,
        #                   keep_initializers_as_inputs=True)
        #
        # print(onnx.checker.check_model(onnx.load(path)))
        #
        # session = onnxruntime.InferenceSession(path)
        # print("session.get_inputs()", session.get_inputs())
        # for o in session.get_inputs():
        #     print(o)
        # for o in session.get_outputs():
        #     print("session.get_outputs()", o)

        # instyle = encoder(F.adaptive_avg_pool2d(I, 256) )

        # img_rec = torch.clamp(img_rec.detach(), -1, 1)
        # viz += [img_rec]
        print('exstyles.keys()', exstyles.keys())
        stylename = list(exstyles.keys())[args.style_id]
        latent = torch.tensor(exstyles[stylename]).to(device)
        if args.preserve_color:
            latent[:, 7:18] = instyle[:, 7:18]
        # extrinsic styte code
        exstyle = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
            latent.shape)

        # load style image if it existsf
        S = None
        if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
            S = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
            # viz += [S]

        # style transfer 
        # input_is_latent: instyle is not in W space
        # z_plus_latent: instyle is in Z+ space
        # use_res: use extrinsic style path, or the style is not transferred
        # interp_weights: weight vector for style combination of two paths
        img_gen, _ = generator([instyle], exstyle, input_is_latent=False, z_plus_latent=True,
                               truncation=args.truncation, truncation_latent=0, use_res=True,
                               interp_weights=args.weight)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        viz += [img_gen]

    print('Generate images successfully!')

    save_name = args.name + '_%d_%s' % (args.style_id, os.path.basename(args.content).split('.')[0])
    save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 4, 2).cpu(),
               os.path.join(args.output_path, save_name + '_overview.jpg'))
    save_image(img_gen[0].cpu(), os.path.join(args.output_path, save_name + '.jpg'))

    print('Save images successfully!')
