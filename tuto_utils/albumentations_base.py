import albumentations


class CFG:
    img_size = 512


transform = albumentations.Compose([
    albumentations.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.9, 1), p=1),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
    albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
    albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
    albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
    albumentations.OneOf([
        albumentations.OpticalDistortion(distort_limit=1.0),
        albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        albumentations.ElasticTransform(alpha=3),
    ], p=0.2),
    albumentations.OneOf([
        albumentations.GaussNoise(var_limit=[10, 50]),
        albumentations.GaussianBlur(),
        albumentations.MotionBlur(),
        albumentations.MedianBlur(),
    ], p=0.2),
    albumentations.Resize(CFG.img_size, CFG.img_size),
    albumentations.OneOf([
        albumentations.ImageCompression(),
        albumentations.Downscale(scale_min=0.1, scale_max=0.15),
    ], p=0.2),
    albumentations.IAAPiecewiseAffine(p=0.2),
    albumentations.IAASharpen(p=0.2),
    albumentations.CoarseDropout(max_height=int(CFG.img_size * 0.1), max_width=int(CFG.img_size * 0.1), max_holes=5,
                                 p=0.5),
])
