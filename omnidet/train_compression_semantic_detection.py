"""
Distance estimation, Semantic segmentation and 2D detection training for OmniDet.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>

# author: Hazem Rashed <hazem.rashed.@valeo.com>

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time
import torch
from colorama import Fore, Style

from losses.detection_loss import ObjectDetectionLoss
from losses.mtl_losses import UncertaintyLoss
from models.detection_decoder import YoloDecoder
from train_detection import DetectionModelBase
from train_semantic_compression import CompressionSemanticModel
from train_semantic import SemanticModel
from train_utils.detection_utils import log_metrics


class CompressionSemanticDetectionModelBase(CompressionSemanticModel):
    def __init__(self, args):
        super().__init__(args)

        self.models["detection"] = YoloDecoder(self.encoder_channels, args=self.args).to(self.device)
        self.parameters_to_train += list(self.models["detection"].parameters())

        self.logs = dict()
        # -- 2D OBJECT DETECTION LOSS --
        self.detection_criterion = ObjectDetectionLoss(config=args)
        self.best_mAP = 0

    def compression_semantic_detection_train(self):
        """Trainer function for compression, semantic and detection prediction"""

        print(f"{Fore.BLUE}=> Initial mAP for detection task: 0{Style.RESET_ALL}")

        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                data_loading_time += (time.time() - before_op_time)
                before_op_time = time.time()

                #use HM codec to compress and decompress image
                # _, inputs["color", 0, 0] = self.codec._run(inputs["color", 0, 0],27,return_rec = True)

                self.inputs_to_device(inputs)

                if self.dl_compress_decompress == True:
                    with torch.no_grad():
                        reconstructed_x = self.compression_model.forward(inputs["color_aug", 0, 0])
                        inputs["color_aug", 0, 0] = reconstructed_x["x_hat"].to(self.device)

                # -- DISTANCE, SEMANTIC SEGMENTATION AND OBJECT DETECTION MODEL PREDICTIONS AND LOSS CALCULATIONS --
                _, outputs, losses = self.compression_semantic_detection_loss_predictions(inputs)

                # -- MTL LOSS --
                losses["mtl_loss"] = self.mtl_loss(losses)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["mtl_loss"].mean().backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["mtl_loss"].mean().cpu().data,
                                  data_loading_time, gpu_time)
                    if self.enable_compression == True:
                        self.compression_statistics("train", inputs, outputs[("compression",0)], losses)
                    SemanticModel.semantic_statistics(self, "train", inputs, outputs, losses)
                    DetectionModelBase.detection_statistics(self, "train")
                    data_loading_time = 0
                    gpu_time = 0

                if self.step % self.args.val_frequency == 0 and self.step != 0:
                    # -- SAVE SEMANTIC MODEL WITH BEST WEIGHTS BASED ON VALIDATION IoU --
                    self.save_best_semantic_weights()
                    # -- SAVE DETECTION MODEL WITH BEST WEIGHTS BASED ON VALIDATION mAP --
                    self.save_best_detection_weights()
                    if self.enable_compression == True:
                        # -- SAVE COMPRESSION MODEL WITH BEST WEIGHTS BASED ON VALIDATION mAP --
                        self.save_best_compression_weights()

                    DetectionModelBase.detection_statistics(self, "val")

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0 and False:
                self.save_model()

        print("Training complete!")

    def compression_semantic_detection_loss_predictions(self, inputs):
        features, outputs, losses = self.compression_semantic_loss_predictions(inputs)
        # Note: We are taking features passed through encoder when the dataset split for all the tasks is same
        outputs, detection_losses = self.predict_detection(inputs, outputs, features=features)
        losses.update(detection_losses)
        if self.args.use_multiple_gpu:
            losses["detection_loss"] = losses["detection_loss"].unsqueeze(0)
        return features, outputs, losses

    def predict_detection(self, inputs, outputs, features=None):
        losses = dict()

        if self.args.pose_model_type == "shared":
            # Use semantic features in MTL instead of encoder features
            detection_output = self.models["detection"](features[0],
                                                        [self.args.input_width, self.args.input_height],
                                                        inputs[("detection_labels", 0)])
        else:
            # Use semantic features in MTL instead of encoder features
            detection_output = self.models["detection"](features,
                                                        [self.args.input_width, self.args.input_height],
                                                        inputs[("detection_labels", 0)])


        outputs[("detection", 0)] = detection_output["yolo_outputs"]

        # -- DETECTION LOSSES --
        detection_losses = self.detection_criterion(detection_output["yolo_output_dicts"],
                                                    detection_output["yolo_target_dicts"])

        losses.update(dict(detection_loss=detection_losses['detection_loss']))

        # -- DETECTION LOGS --
        self.logs.update(log_metrics(detection_output["yolo_output_dicts"],
                                     detection_output["yolo_target_dicts"], detection_losses))
        return outputs, losses

    def save_best_detection_weights(self):
        # 2D Detection validation on each step and save model on improvements.
        precision, recall, AP, f1, ap_class = DetectionModelBase.detection_val(self,
                                                                               iou_thres=0.5,
                                                                               conf_thres=self.args.detection_conf_thres,
                                                                               nms_thres=self.args.detection_nms_thres,
                                                                               img_size=[self.args.input_width,
                                                                                         self.args.input_height])
        if AP.mean() > self.best_mAP:
            print(f"{Fore.BLUE}=> Saving detection model weights with mean_AP of {AP.mean():.3f} "
                  f"at step {self.step} on {self.epoch} epoch.{Style.RESET_ALL}")
            rounded_AP = [round(num, 3) for num in AP]
            print(f"{Fore.BLUE}=> meanAP per class in order: {rounded_AP}{Style.RESET_ALL}")
            self.best_mAP = AP.mean()
            if self.epoch > 50:  # Weights are quite large! Sometimes, life is a compromise.
                self.save_model()
        print(f"{Fore.BLUE}=> Detection val mAP {AP.mean():.3f}{Style.RESET_ALL}")


class CompressionSemanticDetectionModel(CompressionSemanticDetectionModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.mtl_loss = UncertaintyLoss(tasks=self.args.train,enable_compression=self.args.enable_compression).to(self.device)
        self.parameters_to_train += list(self.mtl_loss.parameters())
        self.configure_optimizers()
        self.pre_init()
