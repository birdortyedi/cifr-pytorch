import os
import cv2
import torch
import wandb
import glog as log
import kornia
import numpy as np
import colour

from torch.utils import data
from torchvision import transforms
import torchvision.models as models
from torch.nn import functional as F
from lpips_pytorch import lpips

from modeling.build import build_model, build_discriminators, build_patch_sampler
from modeling.vgg import VGG16FeatLayer
from datasets.transforms import *
from datasets.iffi import IFFIDataset
from losses.consistency import SemanticConsistencyLoss, IDMRFLoss
from losses.adversarial import compute_gradient_penalty
from losses.nce import PatchNCELoss, PatchStyleNCELoss
from utils.data_utils import linear_scaling, linear_unscaling
from metrics.ssim import SSIM


# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        assert self.opt.DATASET.NAME.lower() in ["iffi"]

        self.model_name = "{}_{}".format(self.opt.MODEL.NAME, self.opt.DATASET.NAME) + \
                          "_{}step_{}bs".format(self.opt.TRAIN.NUM_TOTAL_STEP, self.opt.TRAIN.BATCH_SIZE) + \
                          "_{}lr_{}gpu".format(self.opt.MODEL.IFR.SOLVER.LR, self.opt.SYSTEM.NUM_GPU) + \
                          "_{}run".format(self.opt.WANDB.RUN)

        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, resume=self.opt.TRAIN.RESUME, notes=self.opt.WANDB.LOG_DIR, config=self.opt, entity=self.opt.WANDB.ENTITY)

        self.transform = Compose([
            ResizeTwoInstances(self.opt.DATASET.SIZE),
            RandomHorizontalFlipTwoInstances(),
            ToTensor(),
        ])

        self.dataset = IFFIDataset(root=self.opt.DATASET.ROOT, transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.opt.TRAIN.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)

        self.test_transform = Compose([
            ResizeTwoInstances(self.opt.DATASET.SIZE),
            ToTensor(),
        ])

        self.test_dataset = IFFIDataset(root=self.opt.DATASET.TEST_ROOT, transform=self.test_transform)
        self.test_image_loader = data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)

        self.to_pil = transforms.ToPILImage()

        self.net, self.mlp = build_model(self.opt)
        self.discriminator, self.patch_discriminator = build_discriminators(self.opt)
        self.vgg16 = models.vgg16(pretrained=True).features.eval().cuda()
        self.vgg_feat = VGG16FeatLayer(self.vgg16)

        self.optimizer = torch.optim.Adam(self.net.parameters() if self.mlp is None else list(self.net.parameters()) + list(self.mlp.parameters()),
                                          lr=self.opt.MODEL.IFR.SOLVER.LR, betas=self.opt.MODEL.IFR.SOLVER.BETAS)
        self.optimizer_discriminator = torch.optim.Adam(list(self.discriminator.parameters()) + list(self.patch_discriminator.parameters()),
                                                        lr=self.opt.MODEL.D.SOLVER.LR, betas=self.opt.MODEL.D.SOLVER.BETAS)

        self.num_step = self.opt.TRAIN.START_STEP

        self.output_dir = os.path.join(self.opt.TEST.OUTPUT_DIR, self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.SSIM = SSIM()

        if self.opt.TRAIN.START_STEP != 0 and self.opt.TRAIN.RESUME and self.opt.MODEL.CKPT == "":  # find start step from checkpoint file name. TODO
            log.info("Checkpoints loading...")
            self.load_checkpoints(self.opt.TRAIN.START_STEP)
        elif self.opt.MODEL.CKPT != "":
            log.info("Checkpoints loading from ckpt file...")
            self.load_checkpoints_from_ckpt(self.opt.MODEL.CKPT)

        self.check_and_use_multi_gpu()
        self._init_criterion()

    def _init_criterion(self):
        self.reconstruction_loss = torch.nn.L1Loss().cuda()
        self.semantic_consistency_loss = SemanticConsistencyLoss().cuda()
        self.texture_consistency_loss = IDMRFLoss().cuda()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss().cuda()

    def run(self):
        while self.num_step < self.opt.TRAIN.NUM_TOTAL_STEP:
            self.num_step += 1
            info = " [Step: {}/{} ({}%)] ".format(self.num_step, self.opt.TRAIN.NUM_TOTAL_STEP, 100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP)

            imgs, y_imgs, labels = next(iter(self.image_loader))
            imgs = linear_scaling(imgs.float().cuda())
            y_imgs = y_imgs.float().cuda()
            labels = labels.cuda()

            for _ in range(self.opt.MODEL.D.NUM_CRITICS):
                d_loss = self.train_D(imgs, y_imgs)
            info += "D Loss: {} ".format(d_loss)

            g_loss, output = self.train_G(imgs, y_imgs, labels)
            info += "G Loss: {} ".format(g_loss)

            if self.num_step % self.opt.TRAIN.LOG_INTERVAL == 0:
                log.info(info)

            if self.num_step % self.opt.TRAIN.VISUALIZE_INTERVAL == 0:
                idx = self.opt.WANDB.NUM_ROW
                self.wandb.log({"examples": [
                    self.wandb.Image(self.to_pil(y_imgs[idx].cpu()), caption="original_image"),
                    self.wandb.Image(self.to_pil(linear_unscaling(imgs[idx]).cpu()), caption="filtered_image"),
                    self.wandb.Image(self.to_pil(torch.clamp(output, min=0., max=1.)[idx].cpu()), caption="output")
                ]}, commit=False)

            if self.num_step % self.opt.TRAIN.SAVE_INTERVAL == 0 and self.num_step != 0:
                self.do_checkpoint(self.num_step)

            if self.num_step % self.opt.TRAIN.EVAL_INTERVAL == 0 and self.num_step != 0:
                self.evaluate()
            self.wandb.log({})

    def evaluate(self):
        def calc_cie_delta_E(real_path, fake_path):
            real_rgb = cv2.imread(real_path)
            fake_rgb = cv2.imread(fake_path)

            real_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(real_rgb))
            fake_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(fake_rgb))

            delta_E = colour.delta_E(real_lab, fake_lab)
            return np.mean(delta_E)

        self.net.eval()
        log.info(" [Step: {}/{} ({}%)] Evaluating...")
        psnr_lst, ssim_lst, lpips_lst, deltaE_lst = list(), list(), list(), list()
        for batch_idx, (imgs, y_imgs) in enumerate(self.test_image_loader):
            imgs = linear_scaling(torch.cat(imgs, dim=0).float().cuda())
            y_imgs = torch.cat(y_imgs, dim=0).float().cuda()

            with torch.no_grad():
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
                deltaE_batch_lst.append(calc_cie_delta_E(real_path, fake_path))
            deltaE = np.mean(deltaE_batch_lst).item()
            deltaE_lst.append(deltaE)
        results = {"Dataset": self.opt.DATASET.NAME, "PSNR": np.mean(psnr_lst), "SSIM": np.mean(ssim_lst), "LPIPS": np.mean(lpips_lst), "Delta_E": np.mean(deltaE_lst)}
        log.info(results)
        self.wandb.log({"test/psnr": results["PSNR"],
                        "test/ssim": results["SSIM"],
                        "test/lpips": results["LPIPS"],
                        "test/delta_e": results["Delta_E"]}, commit=False)
        self.net.train()

    def train_D(self, x, y):
        self.optimizer_discriminator.zero_grad()

        with torch.no_grad():
            vgg_feat = self.vgg16(x)
        output, _ = self.net(x, vgg_feat)

        real_global_validity = self.discriminator(y).mean()
        fake_global_validity = self.discriminator(output.detach()).mean()
        gp_global = compute_gradient_penalty(self.discriminator, output.data, y.data)

        real_patch_validity = self.patch_discriminator(y).mean()
        fake_patch_validity = self.patch_discriminator(output.detach()).mean()
        gp_fake = compute_gradient_penalty(self.patch_discriminator, output.data, y.data)

        real_validity = real_global_validity + real_patch_validity
        fake_validity = fake_global_validity + fake_patch_validity
        gp = gp_global + gp_fake

        d_loss = -real_validity + fake_validity + self.opt.OPTIM.GP * gp
        d_loss.backward()
        self.optimizer_discriminator.step()

        self.wandb.log({"real_global_validity": -real_global_validity.item(),
                        "fake_global_validity": fake_global_validity.item(),
                        "real_patch_validity": -real_patch_validity.item(),
                        "fake_patch_validity": fake_patch_validity.item(),
                        "gp_global": gp_global.item(),
                        "gp_fake": gp_fake.item(),
                        "real_validity": -real_validity.item(),
                        "fake_validity": fake_validity.item(),
                        "gp": gp.item()}, commit=False)
        return d_loss.item()

    def train_G(self, x, y, labels):
        self.optimizer.zero_grad()

        with torch.no_grad():
            vgg_feat = self.vgg16(x)
        output, aux = self.net(x, vgg_feat)
        labels_pred = self.mlp(aux) if self.mlp is not None else None

        recon_loss = self.reconstruction_loss(output, y)

        with torch.no_grad():
            out_vgg_feat = self.vgg_feat(output)
            y_vgg_feat = self.vgg_feat(y)
        sem_const_loss = self.semantic_consistency_loss(out_vgg_feat, y_vgg_feat)
        tex_const_loss = self.texture_consistency_loss(out_vgg_feat, y_vgg_feat)
        adv_global_loss = -self.discriminator(output).mean()
        adv_patch_loss = -self.patch_discriminator(output).mean()

        if self.mlp is None:
            adv_loss = (adv_global_loss + adv_patch_loss)
        else:
            aux_loss = self.auxiliary_loss(labels_pred, labels.squeeze())
            adv_loss = (adv_global_loss + adv_patch_loss) + self.opt.OPTIM.AUX * aux_loss

        g_loss = self.opt.OPTIM.RECON * recon_loss + \
                 self.opt.OPTIM.SEMANTIC * sem_const_loss + \
                 self.opt.OPTIM.TEXTURE * tex_const_loss * \
                 self.opt.OPTIM.ADVERSARIAL * adv_loss
        g_loss.backward()
        self.optimizer.step()

        self.wandb.log({"recon_loss": recon_loss.item(),
                        "sem_const_loss": sem_const_loss.item(),
                        "tex_const_loss": tex_const_loss.item(),
                        "adv_global_loss": adv_global_loss.item(),
                        "adv_patch_loss": adv_patch_loss.item(),
                        "aux_loss": aux_loss.item() if self.mlp is not None else 0,
                        "adv_loss": adv_loss.item()}, commit=False)
        return g_loss.item(), output.detach()

    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1:
            log.info("Using {} GPUs...".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net).cuda()
            self.mlp = torch.nn.DataParallel(self.mlp).cuda() if self.mlp is not None else self.mlp
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            self.patch_discriminator = torch.nn.DataParallel(self.patch_discriminator).cuda()
        else:
            log.info("GPU ID: {}".format(torch.cuda.current_device()))
            self.net = self.net.cuda()
            self.mlp = self.mlp.cuda() if self.mlp is not None else self.mlp
            self.discriminator = self.discriminator.cuda()
            self.patch_discriminator = self.patch_discriminator.cuda()

    def do_checkpoint(self, num_step):
        if not os.path.exists("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name)):
            os.makedirs("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name), exist_ok=True)

        if self.mlp is not None:
            checkpoint = {
                'num_step': num_step,
                'ifr': self.net.module.state_dict() if isinstance(self.net, torch.nn.DataParallel) else self.net.state_dict(),
                'mlp': self.mlp.module.state_dict() if isinstance(self.mlp, torch.nn.DataParallel) else self.mlp.state_dict(),
                'D': self.discriminator.module.state_dict() if isinstance(self.discriminator, torch.nn.DataParallel) else self.discriminator.state_dict(),
                'patch_D': self.patch_discriminator.module.state_dict() if isinstance(self.patch_discriminator, torch.nn.DataParallel) else self.patch_discriminator.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_D': self.optimizer_discriminator.state_dict()
            }
        else:
            checkpoint = {
                'num_step': num_step,
                'ifr': self.net.module.state_dict() if isinstance(self.net, torch.nn.DataParallel) else self.net.state_dict(),
                'D': self.discriminator.module.state_dict() if isinstance(self.discriminator, torch.nn.DataParallel) else self.discriminator.state_dict(),
                'patch_D': self.patch_discriminator.module.state_dict() if isinstance(self.patch_discriminator, torch.nn.DataParallel) else self.patch_discriminator.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_D': self.optimizer_discriminator.state_dict()
            }
        torch.save(checkpoint, "./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))

    def load_checkpoints(self, num_step):
        checkpoints = torch.load("./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))
        self.num_step = checkpoints["num_step"]
        self.net.load_state_dict(checkpoints["ifr"])
        if "mlp" in checkpoints.keys():
            self.mlp.load_state_dict(checkpoints["mlp"])
        self.discriminator.load_state_dict(checkpoints["D"])
        self.patch_discriminator.load_state_dict(checkpoints["patch_D"])

        self.optimizer.load_state_dict(checkpoints["optimizer"])
        self.optimizer_discriminator.load_state_dict(checkpoints["optimizer_D"])
        self.optimizers_to_cuda()

    def load_checkpoints_from_ckpt(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.num_step = checkpoints["num_step"]
        self.net.load_state_dict(checkpoints["ifr"])
        if "mlp" in checkpoints.keys():
            self.mlp.load_state_dict(checkpoints["mlp"])
        self.discriminator.load_state_dict(checkpoints["D"])
        self.patch_discriminator.load_state_dict(checkpoints["patch_D"])

        self.optimizers_to_cuda()

    def optimizers_to_cuda(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_discriminator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()


class PatchContrastiveTrainer(Trainer):
    def __init__(self, opt):
        self.content_patch_sampler, self.style_patch_sampler = build_patch_sampler(opt)
        self.optimizer_patch = torch.optim.Adam(list(self.content_patch_sampler.parameters()) + list(self.style_patch_sampler.parameters()),
                                                lr=opt.MODEL.PATCH.LR, betas=opt.MODEL.PATCH.BETAS)
        # self.content_patch_sampler, _ = build_patch_sampler(opt)
        # self.optimizer_patch = torch.optim.Adam(self.content_patch_sampler.parameters(), lr=opt.MODEL.PATCH.LR, betas=opt.MODEL.PATCH.BETAS)
        super(PatchContrastiveTrainer, self).__init__(opt)
        log.info("Starting from step {}".format(self.num_step))

    def _init_criterion(self):
        self.reconstruction_loss = torch.nn.L1Loss().cuda()
        self.semantic_consistency_loss = SemanticConsistencyLoss().cuda()
        self.texture_consistency_loss = IDMRFLoss().cuda()
        self.nce_loss_lst = [(PatchNCELoss(self.opt).cuda(), PatchStyleNCELoss(self.opt).cuda()) for _ in range(self.opt.MODEL.PATCH.NUM_LAYERS)]
        # self.nce_loss_lst = [(PatchNCELoss(self.opt).cuda(), None) for _ in range(self.opt.MODEL.PATCH.NUM_LAYERS)]

    def feed(self, x):
        with torch.no_grad():
            feat = self.vgg16(x)
        output, embedding = self.net(x, feat)
        embedding = self.mlp(embedding) if self.mlp is not None else embedding
        return output, embedding

    def calculate_content_NCE_loss(self, feat_q, feat_k):
        feat_k_pool, sample_ids = self.content_patch_sampler(feat_k, self.opt.MODEL.PATCH.NUM_PATCHES, None)
        feat_q_pool, _ = self.content_patch_sampler(feat_q, self.opt.MODEL.PATCH.NUM_PATCHES, sample_ids)

        total_c_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.nce_loss_lst):
            c_nce_loss = crit[0](f_q, f_k)
            total_c_nce_loss += c_nce_loss.mean()

        return total_c_nce_loss / self.opt.MODEL.PATCH.NUM_LAYERS

    def calculate_style_NCE_loss(self, feat_q, feat_k, feat_o):
        feat_k_pool, sample_ids = self.style_patch_sampler(feat_k, self.opt.MODEL.PATCH.NUM_PATCHES, None)
        feat_q_pool, _ = self.style_patch_sampler(feat_q, self.opt.MODEL.PATCH.NUM_PATCHES, sample_ids)
        feat_o_pool, _ = self.style_patch_sampler(feat_o, self.opt.MODEL.PATCH.NUM_PATCHES, sample_ids)

        total_s_nce_loss = 0.0
        for f_q, f_k, f_o, crits in zip(feat_q_pool, feat_k_pool, feat_o_pool, self.nce_loss_lst):
            s_nce_loss = crits[1](f_q, f_k, f_o)
            total_s_nce_loss += s_nce_loss.mean()

        return total_s_nce_loss / self.opt.MODEL.PATCH.NUM_LAYERS

    def calculate_NCE_loss(self, src, tgt, org):
        _, feat_q = self.feed(tgt)  # fake_B
        _, feat_k = self.feed(src)  # real_A
        _, feat_o = self.feed(org)  # real_B

        content_nce_loss = self.calculate_content_NCE_loss(feat_q, feat_k)
        identity_loss = self.calculate_content_NCE_loss(feat_q, feat_o)
        style_nce_loss = self.calculate_style_NCE_loss(feat_q, feat_k, feat_o)

        return content_nce_loss, identity_loss, style_nce_loss

    # real_A: filtered image
    # real_B: original image
    # fake_B: reconstructed image
    def train_G(self, real_A, real_B, labels):
        self.optimizer.zero_grad()
        self.optimizer_patch.zero_grad()

        self.set_D_req_grad(False)

        fake_B, feats = self.feed(real_A)

        recon_loss = self.reconstruction_loss(fake_B, real_B)
        with torch.no_grad():
            out_vgg_feat = self.vgg_feat(fake_B)
            y_vgg_feat = self.vgg_feat(real_B)
        sem_const_loss = self.semantic_consistency_loss(out_vgg_feat, y_vgg_feat)
        tex_const_loss = self.texture_consistency_loss(out_vgg_feat, y_vgg_feat)
        adv_global_loss = -self.discriminator(fake_B).mean()
        adv_patch_loss = -self.patch_discriminator(fake_B).mean()
        adv_loss = (adv_global_loss + adv_patch_loss) * 0.5

        content_nce_loss, identity_loss, style_nce_loss = self.calculate_NCE_loss(real_A, fake_B, real_B)
        # nce_loss = (content_nce_loss + style_nce_loss) * 0.5
        nce_loss = ((content_nce_loss + identity_loss) * 0.5 + style_nce_loss) * 0.5
        # nce_loss = (content_nce_loss + identity_loss) * 0.5

        g_loss = self.opt.OPTIM.CONTRASTIVE * nce_loss + \
                 self.opt.OPTIM.ADVERSARIAL * adv_loss + \
                 self.opt.OPTIM.RECON * recon_loss + \
                 self.opt.OPTIM.SEMANTIC * sem_const_loss + \
                 self.opt.OPTIM.TEXTURE * tex_const_loss

        g_loss.backward()
        self.optimizer_patch.step()
        self.optimizer.step()

        self.wandb.log({
            "recon_loss": recon_loss.item(),
            "sem_const_loss": sem_const_loss.item(),
            "tex_const_loss": tex_const_loss.item(),
            "adv_global_loss": adv_global_loss.item(),
            "adv_patch_loss": adv_patch_loss.item(),
            "adv_loss": adv_loss.item(),
            "contrastive_content_loss": content_nce_loss.item(),
            "contrastive_style_loss": style_nce_loss.item(),
            "contrastive_identity_loss": identity_loss.item(),
            "contrastive_loss": nce_loss.item(),
            "g_loss": g_loss.item()
            }, commit=False)
        return g_loss.item(), fake_B.detach()

    def train_D(self, x, y):
        self.optimizer_discriminator.zero_grad()

        self.set_D_req_grad(True)

        with torch.no_grad():
            vgg_feat = self.vgg16(x)
        output, _ = self.net(x, vgg_feat)

        real_global_validity = F.softplus(self.discriminator(y)).mean()
        fake_global_validity = F.softplus(self.discriminator(output.detach())).mean()

        real_patch_validity = F.softplus(self.patch_discriminator(y)).mean()
        fake_patch_validity = F.softplus(self.patch_discriminator(output.detach())).mean()

        real_validity = (real_global_validity + real_patch_validity) * 0.5
        fake_validity = (fake_global_validity + fake_patch_validity) * 0.5

        gp_global = compute_gradient_penalty(self.discriminator, output.data, y.data)
        gp_fake = compute_gradient_penalty(self.patch_discriminator, output.data, y.data)
        gp = gp_global + gp_fake

        d_loss = (-real_validity + fake_validity) * 0.5 + self.opt.OPTIM.GP * gp
        d_loss.backward()
        self.optimizer_discriminator.step()

        self.wandb.log({"real_global_validity": -real_global_validity.item(),
                        "fake_global_validity": fake_global_validity.item(),
                        "real_patch_validity": -real_patch_validity.item(),
                        "fake_patch_validity": fake_patch_validity.item(),
                        "gp_global": gp_global.item(),
                        "gp_fake": gp_fake.item(),
                        "real_validity": -real_validity.item(),
                        "fake_validity": fake_validity.item(),
                        "gp": gp.item()}, commit=False)
        return d_loss.item()

    def set_D_req_grad(self, req_grad):
        if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1:
            self.discriminator.module.set_requires_grad(req_grad)
            self.patch_discriminator.module.set_requires_grad(req_grad)
        else:
            self.discriminator.set_requires_grad(req_grad)
            self.patch_discriminator.set_requires_grad(req_grad)

    def check_and_use_multi_gpu(self):
        super(PatchContrastiveTrainer, self).check_and_use_multi_gpu()
        self.content_patch_sampler = torch.nn.DataParallel(self.content_patch_sampler).cuda() \
            if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1 else \
            self.content_patch_sampler.cuda()
        self.style_patch_sampler = torch.nn.DataParallel(self.style_patch_sampler).cuda() \
            if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1 else \
            self.style_patch_sampler.cuda()

    def do_checkpoint(self, num_step):
        if not os.path.exists("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name)):
            os.makedirs("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name), exist_ok=True)

        if self.mlp is not None:
            checkpoint = {
                'num_step': num_step,
                'ifr': self.net.module.state_dict() if isinstance(self.net, torch.nn.DataParallel) else self.net.state_dict(),
                'mlp': self.mlp.module.state_dict() if isinstance(self.mlp, torch.nn.DataParallel) else self.mlp.state_dict(),
                'c_patch': self.content_patch_sampler.module.state_dict() if isinstance(self.content_patch_sampler, torch.nn.DataParallel) else self.content_patch_sampler.state_dict(),
                's_patch': self.style_patch_sampler.module.state_dict() if isinstance(self.style_patch_sampler, torch.nn.DataParallel) else self.style_patch_sampler.state_dict(),
                'D': self.discriminator.module.state_dict() if isinstance(self.discriminator, torch.nn.DataParallel) else self.discriminator.state_dict(),
                'patch_D': self.patch_discriminator.module.state_dict() if isinstance(self.patch_discriminator, torch.nn.DataParallel) else self.patch_discriminator.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_patch': self.optimizer_patch.state_dict(),
                'optimizer_D': self.optimizer_discriminator.state_dict()
            }
        else:
            checkpoint = {
                'num_step': num_step,
                'ifr': self.net.module.state_dict() if isinstance(self.net, torch.nn.DataParallel) else self.net.state_dict(),
                'c_patch': self.content_patch_sampler.module.state_dict() if isinstance(self.content_patch_sampler, torch.nn.DataParallel) else self.content_patch_sampler.state_dict(),
                's_patch': self.style_patch_sampler.module.state_dict() if isinstance(self.style_patch_sampler, torch.nn.DataParallel) else self.style_patch_sampler.state_dict(),
                'D': self.discriminator.module.state_dict() if isinstance(self.discriminator, torch.nn.DataParallel) else self.discriminator.state_dict(),
                'patch_D': self.patch_discriminator.module.state_dict() if isinstance(self.patch_discriminator, torch.nn.DataParallel) else self.patch_discriminator.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_patch': self.optimizer_patch.state_dict(),
                'optimizer_D': self.optimizer_discriminator.state_dict()
            }
        torch.save(checkpoint, "./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))

    def load_checkpoints(self, num_step):
        checkpoints = torch.load("./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))
        self.num_step = checkpoints["num_step"]
        self.net.load_state_dict(checkpoints["ifr"])
        if "mlp" in checkpoints.keys():
            self.mlp.load_state_dict(checkpoints["mlp"])
        self.content_patch_sampler.load_state_dict(checkpoints["c_patch"])
        self.style_patch_sampler.load_state_dict(checkpoints["s_patch"])
        self.discriminator.load_state_dict(checkpoints["D"])
        self.patch_discriminator.load_state_dict(checkpoints["patch_D"])

        self.optimizer.load_state_dict(checkpoints["optimizer"])
        self.optimizer_discriminator.load_state_dict(checkpoints["optimizer_D"])
        self.optimizer_patch.load_state_dict(checkpoints['optimizer_patch'])
        self.optimizers_to_cuda()

    def load_checkpoints_from_ckpt(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.num_step = checkpoints["num_step"]
        self.net.load_state_dict(checkpoints["ifr"])
        if self.mlp:  # "mlp" in checkpoints.keys():
            self.mlp.load_state_dict(checkpoints["mlp"])
        if "c_patch" in checkpoints.keys():
            self.content_patch_sampler.load_state_dict(checkpoints["c_patch"])
        if "s_patch" in checkpoints.keys():
            self.style_patch_sampler.load_state_dict(checkpoints["s_patch"])
        self.discriminator.load_state_dict(checkpoints["D"])
        self.patch_discriminator.load_state_dict(checkpoints["patch_D"])

        self.optimizers_to_cuda()

    def optimizers_to_cuda(self):
        super(PatchContrastiveTrainer, self).optimizers_to_cuda()
        for state in self.optimizer_patch.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
