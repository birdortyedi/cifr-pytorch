import argparse
from engine.trainer import PatchContrastiveTrainer
from engine.tester import Tester
from configs.default import get_cfg_defaults

parser = argparse.ArgumentParser()

parser.add_argument("--base_cfg", default="./wandb/run-20201023_213704-3o2q3c4r/config.yaml", metavar="FILE", help="path to config file")
parser.add_argument("--weights", "-w", default="", type=str, help="weights for resuming or pre-trained settings")
parser.add_argument("--dataset", "-d", default="IFFI", help="dataset names: IFFI")
parser.add_argument("--dataset_dir", default="../../Downloads/IFFI-dataset/train", help="dataset directory")
parser.add_argument("--num_step", default=0, help="current step for training")
parser.add_argument("--batch_size", default=8, help="batch size for training")
parser.add_argument("--resume_id", "-r", default="", help="wandb resume for specific id")

parser.add_argument("--test", "-t", action="store_true")

args = parser.parse_args()

if __name__ == '__main__':
    cfg = get_cfg_defaults()

    cfg.TRAIN.RESUME = args.resume_id if args.resume_id else cfg.TRAIN.RESUME
    cfg.DATASET.NAME = args.dataset
    cfg.DATASET.ROOT = args.dataset_dir
    cfg.TRAIN.IS_TRAIN = args.test
    cfg.TRAIN.START_STEP = args.num_step
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.MODEL.CKPT = args.weights
    print(cfg)

    if not args.test:
        trainer = PatchContrastiveTrainer(cfg)
        trainer.run()
    else:
        tester = Tester(cfg)
        tester.eval()
