import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator

# from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

import logging
import math

from ..self_supervised import build_ss_head
from ..roi_heads import build_roi_heads

__all__ = ["SSRCNN"]


@META_ARCH_REGISTRY.register()
class SSRCNN(nn.Module):
    """
    Detection + self-supervised
    """

    def __init__(self, cfg):
        super().__init__()
        # pylint: disable=no-member
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.from_config(cfg)
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.ss_head = build_ss_head(
            cfg, self.backbone.bottom_up.output_shape()
        )

        for i in range(len(self.ss_head)):
            setattr(self, "ss_head_{}".format(i), self.ss_head[i])

        self.to(self.device)

    def from_config(self, cfg):
        # only train/eval the ss branch for debugging.
        self.ss_only = cfg.MODEL.SS.ONLY
        self.feat_level = cfg.MODEL.SS.FEAT_LEVEL  # res4

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        """
        Training methods, which jointly train the detector and the
        self-supervised task.
        """
        if not self.training:
            return self.inference(batched_inputs)
        losses = {}
        accuracies = {}
        # torch.save(batched_inputs, "inputs.pt")
        for i in range(len(self.ss_head)):
            """Using images as the input to SS tasks."""
            head = getattr(self, "ss_head_{}".format(i))
            if head.input != "images":
                continue
            out, tar, ss_losses = head(
                batched_inputs, self.backbone.bottom_up, self.feat_level
            )  # attach new parameters
            losses.update(ss_losses)
            acc = (out.argmax(axis=1) == tar).float().mean().item() * 100
            accuracies["accuracy_ss_{}".format(head.name)] = {
                "accuracy": acc,
                "num": len(tar),
            }

        # for detection part
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        # print(images.tensor.size(), images.image_sizes)
        features = self.backbone(images.tensor)
        # print(features['p2'].size(),features['p3'].size(), features['p4'].size(), features['p5'].size(), features['p6'].size())
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}
        # print(len(proposals), proposals[0])

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        if isinstance(detector_losses, tuple):
            detector_losses, box_features = detector_losses

            for i in range(len(self.ss_head)):
                head = getattr(self, "ss_head_{}".format(i))
                if head.input != "ROI":
                    continue
                # during training, the paired of inputs are put in one batch
                ss_losses, acc = head(box_features)
                losses.update(ss_losses)
                accuracies["accuracy_ss_{}".format(head.name)] = {
                    "accuracy": acc,
                    "num": 1,
                }

        losses.update(detector_losses)
        losses.update(proposal_losses)

        for k, v in losses.items():
            assert math.isnan(v) == False, batched_inputs

        return losses

    def det_inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, others = self.roi_heads(images, features, proposals, None)
            if isinstance(others, tuple):
                others, box_features = others

            else:
                box_features = None
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )
            box_features = None

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results, box_features
        else:
            return results, box_features

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """ used for standard detectron2 test method"""
        results, _ = self.det_inference(
            batched_inputs, detected_instances, do_postprocess
        )
        return results

    def preprocess_image(self, batched_inputs):
        """normalize, pad and batch the input images"""
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images
