"""
Quantitative test script of 2D object detection for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.woodscape_loader import WoodScapeRawDataset
from main import collect_tupperware
from models.detection_decoder import YoloDecoder
from models.resnet import ResnetEncoder
from train_utils.detection_utils import *
from models.compressAI.priors import FactorizedPriorDecoder,ScaleHyperpriorOrg,ScaleHyperpriorDecoder , ScaleHyperpriorNoEntropy , ScaleHyperprior
import torchvision.transforms as T

@torch.no_grad()
def evaluate(args):
    """Function to calculate quantitative results for detection network"""
    val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                      path_file=args.val_file,
                                      is_train=False,
                                      config=args)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=val_dataset.collate_fn)

    print(f"-> Loading model from {args.pretrained_weights}")
    encoder_path = os.path.join(args.pretrained_weights, "encoder.pth")
    decoder_path = os.path.join(args.pretrained_weights, "detection.pth")

    print("=> Loading pretrained encoder")
    encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device)
    loaded_dict_enc = torch.load(encoder_path, map_location=args.device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.eval()

    print("=> Loading pretrained decoder")
    decoder = YoloDecoder(encoder.num_ch_enc, args).to(args.device)
    loaded_dict = torch.load(decoder_path, map_location=args.device)
    decoder.load_state_dict(loaded_dict)
    decoder.eval()

    if args.dl_compress_decompress == True:
        if args.compression_quality == 0:
            compression_model = ScaleHyperprior(128, 192).to(args.device)
            checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q0.pth')
        elif args.compression_quality == 4:
            compression_model = ScaleHyperprior(128, 256).to(args.device)
            checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q4.pth')
        elif args.compression_quality >= 8:
            compression_model = ScaleHyperprior(192, 320).to(args.device)
            checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q8_2023.pth')
        compression_model.load_state_dict(checkpoint)
        #compression_model.update()
        compression_model.eval()

    labels, sample_metrics = (list(), list())  # List of tuples (TP, confs, pred)
    img_size = [feed_width, feed_height]
    for batch_i, inputs in enumerate(tqdm(val_loader, desc="Detecting objects")):

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(args.device)

        if args.dl_compress_decompress == True:
            with torch.no_grad():
                #inputs["color_aug", 0, 0] = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(inputs["color_aug", 0, 0])
                augmenter = T.ColorJitter(#brightness=(0.8, 1.2),
                                       contrast=(1.2))
                                       #saturation=(0.8, 1.2),
                                       #hue=(-0.1, 0.1))
                #inputs["color_aug", 0, 0] = augmenter(inputs["color_aug", 0, 0])

                #out_enc = compression_model.compress(inputs["color_aug", 0, 0])
                #out_dec = compression_model.decompress(out_enc["strings"], out_enc["shape"])
                out_dec = compression_model.forward(inputs["color_aug", 0, 0])
                inputs["color_aug", 0, 0] = out_dec["x_hat"].to(args.device)
                #reconstructed_x = compression_model.forward(inputs["color_aug", 0, 0])
                #inputs["color_aug", 0, 0] = reconstructed_x["x_hat"].to(args.device)

        features = encoder(inputs["color_aug", 0, 0])
        outputs = decoder(features, img_dim=[feed_width, feed_height])["yolo_outputs"]

        # Extract labels
        targets = inputs[("detection_labels", 0)].cpu()
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:6] = xywh2xyxy(targets[:, 2:6])
        targets[:, 2] *= img_size[0]
        targets[:, 3] *= img_size[1]
        targets[:, 4] *= img_size[0]
        targets[:, 5] *= img_size[1]

        outputs = non_max_suppression(outputs, conf_thres=args.detection_conf_thres, nms_thres=args.detection_nms_thres)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.5, args=args)

    # Handle the case of empty sample_metrics:
    if len(sample_metrics) == 0:
        precision, recall, AP, f1, ap_class = 0, 0, 0, 0, 0
    else:
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print(f"AP: {AP}")
    print(f"mAP: {AP.mean()}")
    print(f"recall: {recall.mean()}")
    print(f"precision: {precision.mean()}")
    print(f"f1: {f1.mean()}")


if __name__ == "__main__":
    args = collect_tupperware()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or -1
    evaluate(args)
