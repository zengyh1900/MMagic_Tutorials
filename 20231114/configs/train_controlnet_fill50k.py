default_scope = 'mmagic'

randomness = dict(seed=2022, diff_rank_seed=True)
# env settings
dist_params = dict(backend='nccl')
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# config for model
stable_diffusion_v15_url = 'data/runway_sd15'
controlnet_canny_url = 'data/controlnet_canny'

model = dict(
    type='ControlStableDiffusion',
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    unet=dict(
        type='UNet2DConditionModel',
        subfolder='unet',
        from_pretrained=stable_diffusion_v15_url),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    controlnet=dict(
        type='ControlNetModel',
        # from_pretrained=controlnet_canny_rul
        from_config=controlnet_canny_url  # train from scratch
    ),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor'),
    init_cfg=dict(type='init_from_unet'))

# config for training
train_cfg = dict(by_epoch=False, val_begin=1, max_iters=10000)
optim_wrapper = dict(
    controlnet=dict(optimizer=dict(type='AdamW', lr=1e-5)),
    constructor='MultiOptimWrapperConstructor')

# Config for data loader
pipeline = [
    dict(type='LoadImageFromFile', key='source', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='target', channel_order='rgb'),
    dict(
        type='PackInputs',
        keys=['source', 'target'],
        data_keys='prompt',
        meta_keys=[
            'source_channel_order', 'source_color_type',
            'target_channel_order', 'target_color_type'
        ])
]
dataset = dict(
    type='ControlNetDataset',
    data_root='./data/fill50k',
    ann_file='prompt.json',
    pipeline=pipeline)
train_dataloader = dict(
    dataset=dataset,
    num_workers=16,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    persistent_workers=True,
    batch_size=4)
val_cfg = val_evaluator = val_dataloader = None
test_cfg = test_evaluator = test_dataloader = None

# config for environment
env_cfg = dict(
    # whether to enable cudnn benchmark.
    cudnn_benchmark=True,
    # set multi process parameters.
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters.
    dist_cfg=dict(backend='nccl'))

# set log level
log_level = 'INFO'
log_processor = dict(type='LogProcessor', by_epoch=False)

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = None

# config for model wrapper
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)

# set visualizer
vis_backends = [dict(type='VisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)
# hooks
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=300,
        fixed_input=True,
        # visualize train dataset
        vis_kwargs_list=dict(type='Data', name='fake_img'),
        n_samples=4,
        n_row=2)
]

# configure for default hooks
default_hooks = dict(
    # record time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    # save checkpoint per 10000 iterations
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=20,
        less_keys=['FID-Full-50k/fid', 'FID-50k/fid', 'swd/avg'],
        greater_keys=['IS-50k/is', 'ms-ssim/avg'],
        save_optimizer=True))

# config for environment
env_cfg = dict(
    # whether to enable cudnn benchmark.
    cudnn_benchmark=True,
    # set multi process parameters.
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters.
    dist_cfg=dict(backend='nccl'))

# set log level
log_level = 'INFO'
log_processor = dict(type='LogProcessor', by_epoch=False)

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = None
