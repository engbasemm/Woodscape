"""
Quantitative test script of semantic segmentation for OmniDet.

# usage: ./quantitative_semantic.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_loader.woodscape_loader import WoodScapeRawDataset
from main import collect_tupperware
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder
from utils import AverageMeter


@torch.no_grad()
def evaluate(args):
    """Function to predict for a single image or folder of images"""
    val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                      path_file=args.val_file,
                                      is_train=False,
                                      config=args)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False)

    print(f"-> Loading model from {args.pretrained_weights}")
    encoder_path = os.path.join(args.pretrained_weights, "encoder.pth")
    semantic_decoder_path = os.path.join(args.pretrained_weights, "semantic.pth")

    print("=> Loading pretrained encoder")
    # --- Init model ---
    encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device)
    loaded_dict_enc = torch.load(encoder_path, map_location=args.device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.eval()

    print("=> Loading pretrained decoder")
    decoder = SemanticDecoder(encoder.num_ch_enc, n_classes=args.semantic_num_classes).to(args.device)
    loaded_dict = torch.load(semantic_decoder_path, map_location=args.device)
    decoder.load_state_dict(loaded_dict)
    decoder.eval()

    metric = utils.IoU(args.semantic_num_classes, args.dataset, ignore_index=None)
    acc_meter = AverageMeter()

    for inputs in tqdm(val_loader):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(args.device)

        features = encoder(inputs["color", 0, 0])
        outputs = decoder(features)
        _, predictions = torch.max(outputs["semantic", 0].data, 1)
        metric.add(predictions, inputs["semantic_labels", 0, 0])
        acc, pix = AverageMeter.accuracy(predictions, inputs["semantic_labels", 0, 0])
        acc_meter.update(acc, pix)

    class_iou, mean_iou = metric.value()

    print(f"Mean_IoU: {mean_iou}")
    for k, v in class_iou.items():
        print(f"{k}: {v:.3f}")

    print(f"Accuracy: {acc_meter.average() * 100}")


def collect_args() -> argparse.Namespace:
    """Set command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Config file", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = collect_tupperware()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or -1
    evaluate(args)
