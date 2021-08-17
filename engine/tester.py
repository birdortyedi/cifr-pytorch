import os
import cv2
import json
import torch
import kornia
import colour
import glog as log
import numpy as np
import itertools
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from torch.utils import data
from datasets.transforms import *
import torchvision.models as models
import torchvision.transforms as T
from lpips_pytorch import lpips
from kornia.losses.psnr import PSNRLoss

from datasets.iffi import IFFIDataset
from modeling.build import build_model
from modeling.vgg import VGG16FeatLayer
from metrics.ssim import SSIM
from utils.data_utils import linear_scaling, linear_unscaling


class Tester:
    def __init__(self, opt):
        self.opt = opt
        assert self.opt.DATASET.NAME.lower() in ["iffi"]

        self.classes = ["1997", "Amaro", "Brannan", "Clarendon", "Gingham", "He-Fe", "Hudson", "Lo-Fi", "Mayfair",
                        "Nashville", "Original", "Perpetua", "Sutro", "Toaster", "Valencia", "Willow", "X-Pro II"]

        self.model_name = "{}_{}".format(self.opt.MODEL.NAME, self.opt.DATASET.NAME) + \
                          "_{}step_{}bs".format(self.opt.TRAIN.NUM_TOTAL_STEP, self.opt.TRAIN.BATCH_SIZE) + \
                          "_{}lr_{}gpu".format(self.opt.MODEL.IFR.SOLVER.LR, self.opt.SYSTEM.NUM_GPU) + \
                          "_{}run".format(self.opt.WANDB.RUN)
        self.output_dir = os.path.join(self.opt.TEST.OUTPUT_DIR, self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.transform = Compose([
            ResizeTwoInstances(self.opt.DATASET.SIZE),
            ToTensor(),
        ])
        self.to_pil = T.ToPILImage()

        self.dataset = IFFIDataset(root=self.opt.DATASET.TEST_ROOT, transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)

        self.net, self.mlp = build_model(self.opt)
        self.vgg16 = models.vgg16(pretrained=True).features.eval().cuda()
        self.vgg_feat = VGG16FeatLayer(self.vgg16)

        self.PSNR = PSNRLoss(max_val=1.)
        self.SSIM = SSIM()  # kornia's SSIM is buggy.

        self.check_and_use_multi_gpu()

        self.load_checkpoints_from_ckpt(self.opt.MODEL.CKPT)
        self.net.eval()

    def load_checkpoints_from_ckpt(self, ckpt_path):
        log.info("Checkpoints ({}) loading...".format(ckpt_path))
        checkpoints = torch.load(ckpt_path)
        self.net.load_state_dict(checkpoints["ifr"])

    def check_and_use_multi_gpu(self):
        log.info("GPU ID: {}".format(torch.cuda.current_device()))
        self.net = self.net.cuda()

    def eval(self):
        psnr_lst, ssim_lst, lpips_lst, deltaE_lst = list(), list(), list(), list()
        with torch.no_grad():
            # all_preds, all_targets = torch.tensor([]), torch.tensor([])
            for batch_idx, (imgs, y_imgs) in enumerate(self.image_loader):
                imgs = linear_scaling(torch.cat(imgs, dim=0).float().cuda())
                y_imgs = torch.cat(y_imgs, dim=0).float().cuda()

                vgg_feat = self.vgg16(imgs)
                output, _ = self.net(imgs, vgg_feat)
                output = torch.clamp(output, max=1., min=0.)

                ssim = self.SSIM(255. * y_imgs, 255. * output).item()
                ssim_lst.append(ssim)

                psnr = -kornia.psnr_loss(output, y_imgs, max_val=1.).item()  # -self.PSNR(y_imgs, output).item()
                psnr_lst.append(psnr)

                lpps = lpips(y_imgs, output, net_type='alex', version='0.1').item() / len(y_imgs)  # TODO ?? not sure working
                lpips_lst.append(lpps)

                os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
                deltaE_batch_lst = list()
                for i, (y_img, img, out) in enumerate(zip(y_imgs.cpu(), linear_unscaling(imgs).cpu(), output.cpu())):
                    real_path = os.path.join(self.output_dir, "images", "{}_{}_real_A.png".format(batch_idx, i))
                    fake_path = os.path.join(self.output_dir, "images", "{}_{}_real_B.png".format(batch_idx, i))
                    filtered_path = os.path.join(self.output_dir, "images", "{}_{}_fake_B.png".format(batch_idx, i))
                    self.to_pil(y_img).save(real_path)
                    self.to_pil(img).save(filtered_path)
                    self.to_pil(out).save(fake_path)
                    deltaE_batch_lst.append(self.calc_cie_delta_E(real_path, fake_path))
                deltaE = np.mean(deltaE_batch_lst).item()
                deltaE_lst.append(deltaE)
                log.info("{}/{}\tLPIPS: {}\tSSIM: {}\tPSNR: {}\t Delta E: {}"
                         "".format(batch_idx + 1, len(self.image_loader), round(lpps, 3), round(ssim, 3), round(psnr, 3), round(deltaE, 3)))

        results = {"Dataset": self.opt.DATASET.NAME, "PSNR": np.mean(psnr_lst), "SSIM": np.mean(ssim_lst), "LPIPS": np.mean(lpips_lst), "Delta_E": np.mean(deltaE_lst)}
        log.info(results)

        # with open(os.path.join(self.output_dir, "metrics.json"), "a+") as f:
        #     json.dump(results, f)

    @staticmethod
    def calc_cie_delta_E(real_path, fake_path):
        real_rgb = cv2.imread(real_path)
        fake_rgb = cv2.imread(fake_path)

        real_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(real_rgb))
        fake_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(fake_rgb))

        delta_E = colour.delta_E(real_lab, fake_lab)
        return np.mean(delta_E)
