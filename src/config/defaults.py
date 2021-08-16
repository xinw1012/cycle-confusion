""" Default Arguments """
from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C


# Load options that differ from the original detectron2 defaults. 

# ---------------------------------------------------------------------------- #
# Data Augmentation
# ---------------------------------------------------------------------------- #
# disable random flip during training
_C.DATALOADER.NO_FLIP = False
# offset range for pairs during pair sampling
_C.DATALOADER.PAIR_OFFSET_RANGE = 1


# ---------------------------------------------------------------------------- #
# Self supervised options
# ---------------------------------------------------------------------------- #
_C.MODEL.SS = CN()
_C.MODEL.SS.NAME = [
    "build_rotation_head"
]  # to be compatible with the exisiting configs; for more than one ss task, add bracket here
_C.MODEL.SS.FEAT_LEVEL = "res4"
_C.MODEL.SS.NUM_CLASSES = 4
_C.MODEL.SS.CROP_SIZE = 224
_C.MODEL.SS.LOSS_SCALE = 0.1
_C.MODEL.SS.RATIO = 1.0
_C.MODEL.SS.CLASS_FILE = "permutations/"
_C.MODEL.SS.ONLY = False
_C.MODEL.SS.JIGSAW = CN()
_C.MODEL.SS.JIGSAW.NORM = False
_C.MODEL.SS.COEF = -1.0
_C.MODEL.SS.ROI_THR = 0.8  #set 0.8 for training
_C.MODEL.SS.ROI_ALL = False  # use all ROI without score filtering and nms
_C.MODEL.SS.ENABLE_BATCH = False 
_C.MODEL.SS.BATCH_SIZE = 32


# ---------------------------------------------------------------------------- #
# Specific testing time training options
# ---------------------------------------------------------------------------- #
# _C.TTT = CN()
# _C.TTT.ENABLE = False
# _C.TTT.STEPS = 10
# _C.TTT.SAVE_ROI = False
# _C.TTT.MAX_ITERS = 10
# _C.TTT.BATCH_SIZE = 32
# _C.TTT.RANDOM_BATCH = False
# _C.TTT.INTERVAL = 1
# _C.TTT.ENABLE_BATCH = False
# _C.TTT.NO_BP = False
# _C.TTT.SAVE_BN = False
# _C.TTT.USE_BN = False

# # use full ROI proposal during testing
# _C.TTT.ROI_THR = 0.8  # use 0.8 for training
# _C.TTT.ROI_ALL = False  # use all ROI without score filtering and nms


# _C.TTT.REVERSE = False
# _C.TTT.SS_THRESHOLD = 0.0
# _C.TTT.EXTRA_STEPS = 0
# _C.TTT.WARMUP_ITERS = 0
# _C.TTT.ORACLE = False
# _C.TTT.LAST_OCC = False
# _C.TTT.ADAPT = False
# _C.TTT.ORACLE = False
# _C.TTT.RESET = False
# _C.TTT.CLASS_WEIGHT = False
# _C.TTT.ALL_WEIGHT = False

# _C.CONST = CN()
# _C.CONST.TOPK = 1
# _C.CONST.STEP = 1

