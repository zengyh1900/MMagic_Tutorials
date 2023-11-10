

# config for model
stable_diffusion_v15_url = 'data/runway_sd15'
default_scope = 'mmagic'

val_prompts = [
    'a sks dog in basket', 'a sks dog on the mountain',
    'a sks dog beside a swimming pool', 'a sks dog on the desk',
    'a sleeping sks dog', 'a screaming sks dog', 'a man in the garden'
]

lora_config = dict(target_modules=['to_q', 'to_k', 'to_v'])
model = dict(
    type='DreamBooth',
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
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor', data_keys=None),
    prior_loss_weight=0,
    val_prompts=val_prompts,
    lora_config=lora_config
    )

# config for training
train_cfg = dict(by_epoch=False, val_begin=1, val_interval=10000,max_iters=1000)


optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    # Only optimize LoRA mappings
    modules='.*.lora_mapping',
    # NOTE: lr should be larger than dreambooth finetuning
    optimizer=dict(type='AdamW', lr=5e-4),
    accumulative_counts=1)

pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='PackInputs')
]
dataset = dict(
    type='DreamBoothDataset',
    data_root='./data/dog6',
    # TODO: rename to instance
    concept_dir='imgs',
    prompt='a photo of sks dog',
    pipeline=pipeline)
train_dataloader = dict(
    dataset=dataset,
    num_workers=16,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    persistent_workers=True,
    batch_size=1)
val_cfg = val_evaluator = val_dataloader = None
test_cfg = test_evaluator = test_dataloader = None

# hooks
# configure for default hooks
default_hooks = dict(
    # record time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook',interval=10,log_metric_by_epoch=False),
    # save checkpoint per 10000 iterations
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=20,
        less_keys=['FID-Full-50k/fid', 'FID-50k/fid', 'swd/avg'],
        greater_keys=['IS-50k/is', 'ms-ssim/avg'],
        save_optimizer=True))


randomness = dict(seed=2022, diff_rank_seed=True)

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

# config for model wrapperq
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)
