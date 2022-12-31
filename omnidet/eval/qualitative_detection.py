"""
Qualitative test script of 2D detection for OmniDet.

# usage: ./qualitative_detection.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os
import math
import cv2
import yaml
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchmetrics import PeakSignalNoiseRatio
#from eval.qualitative_semantic import pre_image_op
from main import collect_args
from models.detection_decoder import YoloDecoder
from models.resnet import ResnetEncoder
from train_utils.detection_utils import *
from utils import Tupperware
from models.compressAI.priors import FactorizedPriorDecoder,ScaleHyperpriorOrg,ScaleHyperpriorDecoder , ScaleHyperpriorNoEntropy , ScaleHyperprior
import torchvision.transforms as T

FRAME_RATE = 1


def pre_image_op(args, index, frame_index, cam_side ):
    total_car1_images = 6054
    cropped_coords = dict(Car1=dict(FV=(114, 110, 1176, 610),
                                    MVL=(343, 5, 1088, 411),
                                    MVR=(185, 5, 915, 425),
                                    RV=(186, 203, 1105, 630)),
                          Car2=dict(FV=(160, 272, 1030, 677),
                                    MVL=(327, 7, 1096, 410),
                                    MVR=(175, 4, 935, 404),
                                    RV=(285, 187, 1000, 572)))
    if args.crop:
        if int(frame_index[1:]) < total_car1_images:
            cropped_coords = cropped_coords["Car1"][cam_side]
        else:
            cropped_coords = cropped_coords["Car2"][cam_side]
    else:
        cropped_coords = None

    return get_image(args, index, cropped_coords, frame_index, cam_side )
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def get_image(args, index, cropped_coords, frame_index, cam_side ):
    recording_folder = args.rgb_images if index == 0 else "previous_images"
    file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
    path = os.path.join(args.dataset_dir, recording_folder, file)
    path_q37 = os.path.join(args.dataset_dir, "rgb_images_Q37", file)
    path_q32 = os.path.join(args.dataset_dir, "rgb_images_Q32", file)
    image = Image.open(path).convert('RGB')
    image_q37 = Image.open(path_q37).convert('RGB')
    image_q32 = Image.open(path_q32).convert('RGB')

    if args.crop:
        return image.crop(cropped_coords) , image_q32.crop(cropped_coords) , image_q37.crop(cropped_coords)
    return image , image_q32 , image_q37

def color_encoding_woodscape_detection():
    detection_classes = dict(vehicles=(43, 125, 255), rider=(255, 0, 0), person=(216, 45, 128),
                             traffic_sign=(255, 175, 58), traffic_light=(43, 255, 255))
    detection_color_encoding = np.zeros((5, 3), dtype=np.uint8)
    for i, (k, v) in enumerate(detection_classes.items()):
        detection_color_encoding[i] = v
    return detection_color_encoding

@torch.no_grad()
def test_simple(args):
    """Function to predict for a single image or folder of images"""
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    detection_color_encoding = color_encoding_woodscape_detection()
    encoder_path = os.path.join(args.pretrained_weights, "encoder.pth")
    depth_decoder_path = os.path.join(args.pretrained_weights, "detection.pth")

    print("=> Loading pretrained encoder")
    encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device)
    loaded_dict_enc = torch.load(encoder_path, map_location=args.device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.eval()

    print("=> Loading pretrained decoder")
    decoder = YoloDecoder(encoder.num_ch_enc, args).to(args.device)
    loaded_dict = torch.load(depth_decoder_path, map_location=args.device)
    decoder.load_state_dict(loaded_dict)
    decoder.eval()

    if args.dl_compress_decompress == True:
        if args.compression_quality == 0:
            compression_model = ScaleHyperprior(128, 192) .to(args.device)
            checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q0.pth')
        elif args.compression_quality == 4:
            compression_model = ScaleHyperprior(128, 256).to(args.device)
            checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q4.pth')
        elif args.compression_quality >= 8:
            compression_model = ScaleHyperprior(192, 320) .to(args.device)
            checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q8_2023.pth')
        compression_model.load_state_dict(checkpoint)
        #compression_model.update()
        compression_model.eval()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.join(args.output_directory, f"{args.video_name}.mp4")
    video = cv2.VideoWriter(video_name, fourcc, FRAME_RATE, (feed_width, feed_height))

    image_paths = [line.rstrip('\n') for line in open(args.val_file)]
    print(f"=> Predicting on {len(image_paths)} validation images")

    for idx, image_path in enumerate(image_paths):
        if image_path.endswith(f"_detection.png"):
            continue
        frame_index, cam_side = image_path.split('.')[0].split('_')
        input_image , input_image_q32 , input_image_q37 = pre_image_op(args, 0, frame_index, cam_side )

        input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(args.device)

        input_image_q32 = input_image_q32.resize((feed_width, feed_height), Image.LANCZOS)
        input_image_q32 = transforms.ToTensor()(input_image_q32).unsqueeze(0)
        input_image_q32 = input_image_q32.to(args.device)

        input_image_q37 = input_image_q37.resize((feed_width, feed_height), Image.LANCZOS)
        input_image_q37 = transforms.ToTensor()(input_image_q37).unsqueeze(0)
        input_image_q37 = input_image_q37.to(args.device)

        if args.dl_compress_decompress == True:
            with torch.no_grad():
                '''policy = T.AutoAugmentPolicy.IMAGENET
                augmenter = T.AutoAugment(policy)
                input_image = augmenter(input_image)'''
                augmenter = T.ColorJitter(#brightness=(0.8, 1.2),
                                       contrast=(0.5))
                                       #saturation=(0.8, 1.2),
                                       #hue=(-0.1, 0.1))
                #input_image = augmenter(input_image)
                '''out_enc = compression_model.compress(input_image)
                out_dec = compression_model.decompress(out_enc["strings"], out_enc["shape"])

                torch.clamp(out_dec["x_hat"], min=0, max=1 )
                x = input_image.squeeze(0)
                num_pixels = x.size(0) * x.size(1) * x.size(2)
                bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                print(" - psnr:", round(psnr(input_image, out_dec["x_hat"]), 5),
                " - ms-ssim:", round(ms_ssim(input_image, out_dec["x_hat"], data_range=1.0).item(), 5),
                "compressed size:", sum(len(s[0]) for s in out_enc["strings"]) , "bpp:",bpp)'''

                out_dec = compression_model.forward(input_image)
                x = input_image.squeeze(0)
                num_pixels = x.size(0) * x.size(1) * x.size(2)

                bpp = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in out_dec["likelihoods"].values()
                )
                bpp = round ( bpp.item() , 5)
                psnr_value = round(psnr(x, out_dec["x_hat"]) , 5)
                psnr_value_q32 = round(psnr(x , input_image_q32.squeeze(0)), 5)
                psnr_value_q37 = round(psnr(x , input_image_q37.squeeze(0)), 5)
                mse = F.mse_loss(input_image, out_dec["x_hat"]).item()
                print(" psnr_dl: ",psnr_value,"- psnr_q32:", psnr_value_q32,"- psnr_q37:", psnr_value_q37 ," - bpp :" ,bpp ,
                      " - ms-ssim:", round(ms_ssim(input_image, out_dec["x_hat"], data_range=1.0).item(), 5),
                      " - ms-ssim_q37:", round(ms_ssim(input_image_q37,input_image, data_range=1.0).item(), 5),
                      " - ms-ssim_q32:", round(ms_ssim(input_image_q32, input_image, data_range=1.0).item(), 5)
                      )


                input_image = out_dec["x_hat"].to(args.device)
                #reconstructed_x = compression_model.forward(input_image)
                #input_image = reconstructed_x["x_hat"].to(args.device)

        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_im = os.path.join(args.output_directory, f"{output_name}_detection.png")

        # PREDICTION

        features = encoder(input_image)
        outputs = decoder(features, img_dim=[feed_width, feed_height])
        outputs = non_max_suppression(outputs["yolo_outputs"],
                                      conf_thres=args.detection_conf_thres,
                                      nms_thres=args.detection_nms_thres)

        img_d = input_image[0].cpu().detach().numpy()
        img_d = np.transpose(img_d, (1, 2, 0))
        img_cpu = np.zeros(img_d.shape, img_d.dtype)
        img_cpu[:, :, :] = img_d[:, :, :] * 255

        if not outputs[0] is None:
            outputs = torch.cat(outputs, dim=0)
            for box in outputs:
                # Get class name and color
                cls_pred = int(box[6])
                class_color = (detection_color_encoding[cls_pred]).tolist()
                x1, y1, conf = box[0], box[1], box[4]
                box = get_contour([box[0], box[1], box[2], box[3]], box[5]).exterior.coords
                boxes = np.int0(box)[0:4]
                box = np.int0([[b[0], b[1]] for b in boxes])
                cv2.drawContours(img_cpu, [box], 0, class_color, thickness=2)
                cv2.putText(img_cpu, str(f"{conf:.2f}"), (np.uint16(x1) - 5, np.uint16(y1)  - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5e-3 * img_cpu.shape[0], (0, 255, 0), 1)

            video.write(cv2.cvtColor(np.uint8(img_cpu), cv2.COLOR_RGB2BGR))
            cv2.imwrite(name_dest_im, cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR))

        print(f"=> Processed {idx + 1} of {len(image_paths)} images - saved prediction to {name_dest_im}")

    video.release()
    print(f"=> LoL! beautiful video created and dumped to disk. \n"
          f"=> Done!")


if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    test_simple(args)
