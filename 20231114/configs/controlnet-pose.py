# config for model
stable_diffusion_v15_url = 'data/runway_sd15'
controlnet_canny_url = 'data/controlnet_openpose'
custom_model = 'data/Counterfeit-V2.5'

model = dict(
    type='ControlStableDiffusion',
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=custom_model,
        subfolder='vae'),
    unet=dict(
        type='UNet2DConditionModel',
        subfolder='unet',
        from_pretrained=custom_model),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    controlnet=dict(
        type='ControlNetModel', from_pretrained=controlnet_canny_url),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor'),
    init_cfg=dict(type='convert_from_unet'))
