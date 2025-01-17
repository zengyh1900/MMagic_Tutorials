experiment_name = 'inst-colorization_full_official_cocostuff_256x256'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

stage = 'full'

model = dict(
    type='InstColorization',
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
    image_model=dict(
        type='ColorizationNet', input_nc=4, output_nc=2, norm_type='batch'),
    instance_model=dict(
        type='ColorizationNet', input_nc=4, output_nc=2, norm_type='batch'),
    fusion_model=dict(
        type='FusionNet', input_nc=4, output_nc=2, norm_type='batch'),
    color_data_opt=dict(
        ab_thresh=0,
        p=1.0,
        sample_PS=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ],
        ab_norm=110,
        ab_max=110.,
        ab_quant=10.,
        l_norm=100.,
        l_cent=50.,
        mask_cent=0.5),
    which_direction='AtoB',
    loss=dict(type='HuberLoss', delta=.01))

# yapf: disable
test_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(
        type='InstanceCrop',
        config_file='mmdet::mask_rcnn/mask-rcnn_x101-32x8d_fpn_ms-poly-3x_coco.py',  # noqa
        from_pretrained='data/inst-colorizatioon_full_official_cocostuff-256x256-5b9d4eee.pth',
        finesize=256,
        box_num_upbound=5),
    dict(
        type='Resize',
        keys=['img', 'cropped_img'],
        scale=(256, 256),
        keep_ratio=False),
    dict(
        type='PackInputs',
        data_keys=['box_info', 'box_info_2x', 'box_info_4x', 'box_info_8x']),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        out_dir=save_dir,
        by_epoch=False,
        max_keep_ckpts=10,
        save_best='PSNR',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)

load_from = None
resume = False

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
