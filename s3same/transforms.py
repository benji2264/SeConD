from lightly.transforms import (
    VICRegViewTransform,
    DINOViewTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)

GLOBAL_SIZE = 128
LOCAL_SIZE = 64

byol_transform = [
    BYOLView1Transform(input_size=GLOBAL_SIZE),
    BYOLView2Transform(input_size=GLOBAL_SIZE),
]

vicreg_transform = VICRegViewTransform(input_size=GLOBAL_SIZE, cj_strength=0.5)

# Redefine the others with DINO transforms, which is a more generic class.
# See lightly.transforms to understand the different parameters
simclr_global = DINOViewTransform(
    crop_size=GLOBAL_SIZE,
    cj_strength=0.5,
    crop_scale=(0.5, 1.0),
    cj_sat=0.8,
    gaussian_blur=0.5,
    solarization_prob=0.0,
)

simclr_local = DINOViewTransform(
    crop_size=LOCAL_SIZE,
    cj_strength=0.5,
    crop_scale=(0.2, 0.5),
    cj_sat=0.8,
    gaussian_blur=0.5,
    solarization_prob=0.0,
)

dino_global = DINOViewTransform(
    crop_size=GLOBAL_SIZE,
    crop_scale=(0.4, 1.0),
    cj_strength=0.5,
)

dino_local = DINOViewTransform(
    crop_size=LOCAL_SIZE,
    crop_scale=(0.05, 0.4),
    cj_strength=0.5,
)

swav_global = DINOViewTransform(
    crop_scale=(0.14, 1.0),
    crop_size=GLOBAL_SIZE,
    cj_strength=0.5,
    cj_sat=0.8,
    gaussian_blur=0.5,
    solarization_prob=0.0,
)

swav_local = DINOViewTransform(
    crop_scale=(0.05, 0.14),
    crop_size=LOCAL_SIZE,
    cj_strength=0.5,
    cj_sat=0.8,
    gaussian_blur=0.5,
    solarization_prob=0.0,
)

model_transforms = {
    "SimCLR": [simclr_global, simclr_local],
    "MoCo": [simclr_global, simclr_local],
    "DINO": [dino_global, dino_local],
    "BYOL": [byol_transform, None],
    "VICReg": [vicreg_transform, None],
    "BarlowTwins": [byol_transform, None],
}
