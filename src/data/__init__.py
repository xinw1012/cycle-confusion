from .builtin import (
    register_all_bdd_tracking,
    register_all_waymo,
)

from .pair_sampler import PairTrainingSampler, PairDataLoader

# from .common import MapDataset

from .build import build_detection_train_loader, get_detection_dataset_dicts

# Register them all under "./datasets"
# register_all_bdd100k()
register_all_bdd_tracking()
register_all_waymo()
