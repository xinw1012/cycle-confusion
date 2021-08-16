import json
import os
import os.path as osp
from collections import defaultdict

from detectron2.data import DatasetCatalog
from .coco import register_coco_instances  # use customized data register

__all__ = [
    "register_all_bdd_tracking",
    "register_all_waymo",
]


def load_json(filename):
    with open(filename, "r") as fp:
        reg_file = json.load(fp)
    return reg_file


# ==== Predefined datasets and splits for BDD100K ==========
# BDD100K MOT set domain splits.
_PREDEFINED_SPLITS_BDDT = {
    "bdd_tracking_2k": {
        "bdd_tracking_2k_train": (
            "bdd100k/images/track/train",
            "bdd100k/labels/track/bdd100k_mot_train_coco.json",
        ),
        "bdd_tracking_2k_val": (
            "bdd100k/images/track/val",
            "bdd100k/labels/track/bdd100k_mot_val_coco.json",
        ),
    },
    "bdd_tracking_2k_3cls": {
        "bdd_tracking_2k_train_3cls": (
            "bdd100k/images/track/train",
            "bdd100k/labels/track_3cls/bdd100k_mot_train_coco.json"
        ),
        "bdd_tracking_2k_val_3cls": (
            "bdd100k/images/track/val",
            "bdd100k/labels/track_3cls/bdd100k_mot_val_coco.json"
        ),
    },
}

# Register data for different domains as well as different sequence.
domain_path = "bdd100k/labels/box_track_20/domain_splits/"
train_splits = load_json(
    osp.join("datasets", domain_path, "bdd100k_mot_domain_splits_train.json")
)
val_splits = load_json(
    osp.join("datasets", domain_path, "bdd100k_mot_domain_splits_val.json")
)


# per_seq_{split}_{key}_{_attr}: [dataset_names]
per_seq_maps = defaultdict(list)

# register the BDD100K per domain sets
for split, result in [("train", train_splits), ("val", val_splits)]:
    for key, values in result.items():
        # key is ["timeofday", "scene", "weather"]
        for attr, seqs in values.items():
            # attr is the actual attribute under each category like
            # `daytime`, `night`, etc. Values are list of sequence names.
            if "/" in attr or " " in attr:
                if "/" in attr:
                    _attr = attr.replace("/", "-")
                if " " in attr:
                    _attr = attr.replace(" ", "-")
            else:
                _attr = attr

            # register per domain values.
            for suffix in ["", "_3cls"]:
                _PREDEFINED_SPLITS_BDDT["bdd_tracking_2k{}".format(suffix)][
                    "bdd_tracking_2k_{}_{}{}".format(split, _attr, suffix)
                ] = (
                    "bdd100k/images/track/{}".format(split),
                    osp.join(
                        domain_path.replace("box_track_20",
                                            "box_track_20{}".format(suffix)),
                        "labels",
                        split,
                        "{}_{}_{}_coco.json".format(split, key, _attr),
                    ),
                )


def register_all_bdd_tracking(root="datasets"):
    # bdd_tracking meta data
    # fmt: off
    thing_classes = ['pedestrian', 'rider', 'car', 'truck', 'bus', 
    'train', 'motorcycle', 'bicycle']
    # thing_classes = ['person', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
    thing_classes_3cls = ["vehicle", "pedestrian", "cyclist"]
    # fmt: on
    for DATASETS in [_PREDEFINED_SPLITS_BDDT]:
        for key, value in DATASETS.items():
            metadata = {
                "thing_classes": thing_classes_3cls
                if "3cls" in key
                else thing_classes
            }
            for name, (img_dir, label_file) in value.items():
                register_coco_instances(
                    name,
                    metadata,
                    os.path.join(root, label_file),
                    os.path.join(root, img_dir),
                )


# ==== Predefined datasets and splits for Waymo ==========
_PREDEFINED_SPLITS_WAYMO = {"waymo": {}}
for direc in ['all', 'front', 'front_left', 'front_right', 'side_left', 'side_right']:
    for mode in ['train', 'val', 'test']:
        _PREDEFINED_SPLITS_WAYMO['waymo']['waymo_%s_%s' % (direc, mode,)] = (
            'waymo/images', 'waymo/labels/waymo12_%s_%s_3cls.json' % (direc, mode,),
        )


def register_all_waymo(root='datasets'):
    # waymo meta data
    # fmt: off
    thing_classes = ['vehicle', 'pedestrian', 'cyclist']
    # fmt: on
    metadata = {"thing_classes": thing_classes}
    for d in [_PREDEFINED_SPLITS_WAYMO]:
        for key, value in d.items():
            for name, (img_dir, label_file) in value.items():
                register_coco_instances(name, metadata,
                                        os.path.join(root, label_file),
                                        os.path.join(root, img_dir))
