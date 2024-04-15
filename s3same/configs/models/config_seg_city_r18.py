_base_ = "mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb2-80k_cityscapes-512x1024.py"

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model = dict(
    pretrained="~/scratch/models/r18_city_renamed.pth",
    backbone=dict(type="ResNet", depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64),
)
train_dataloader = dict(
    batch_size=8,
)
