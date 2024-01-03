import _thread
import argparse
import contextlib
import datetime
import heapq
import logging
import os
import shutil
import tempfile
import threading
import time
import random
import copy
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from modelscope import snapshot_download
import torch.utils.checkpoint as checkpoint
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverSinglestepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL, 
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils import is_wandb_available
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPTokenizer
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional, Tuple, Union


import reward_optimization.prompts
import reward_optimization.rewards
from reward_optimization.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from reward_optimization.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob, step, pipeline_with_logprob_with_grad
from merge_lora import merge_lora
from utils import AdaptiveKLController

# set max rl wall-clock time
max_rl_time = int(float(1) * 60 * 60)
os.environ["MAX_RL_TIME"] = str(max_rl_time)



def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)



class TimeoutException(Exception):
    def __init__(self, msg=""):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=""):
    """The context manager to limit the execution time of a function call given `seconds`.
    Borrowed from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call.
    """
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="The top-level logging directory for checkpoint saving.",
    )
    parser.add_argument(
        "--cache_log_file",
        type=str,
        default="train_kohya_log.txt",
        help="The output log file path. Use the same log file as train_lora.py",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train for. Each epoch is one round of sampling from the model "
        "followed by training on those samples.",
    )
    parser.add_argument("--save_freq", type=int, default=20, help="Number of epochs between saving model checkpoints.")
    parser.add_argument(
        "--num_checkpoint_limit",
        type=int,
        default=5,
        help="Number of checkpoints to keep before overwriting old ones.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), "
        "or a directory containing checkpoints, in which case the latest one will be used. "
        "`args.use_lora` must be set to the same value as the run that generated the saved checkpoint.",
    )

    # Sampling
    parser.add_argument(
        "--sample_num_steps",
        type=int,
        default=40,
        help="Number of sampler inference steps.",
    )
    parser.add_argument(
        "--sample_guidance_scale", type=int, default=7, help="A guidance_scale during training for sampling."
    )
    parser.add_argument(
        "--sample_eta",
        type=float,
        default=1.0,
        help="The amount of noise injected into the DDIM sampling process.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="The batch size (per GPU!) to use for sampling.",
    )
    parser.add_argument(
        "--sample_num_batches_per_epoch",
        type=int,
        default=2,
        help="Number of batches to sample per epoch. The total number of samples per epoch is "
        "`sample_num_batches_per_epoch * sample_batch_size * num_gpus.`",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models. (SD base model config.)"
    )
    # parser.add_argument( 
    #     "--pretrained_model_ckpt",
    #     type=str,
    #     required=True,
    #     help="Path to pretrained model or model identifier from huggingface.co/models.",
    # )
    parser.add_argument(
        "--face_lora_path",
        type=str,
        required=True,
        help="Path to the face lora model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--sub_path",
        type=str,
        default=None,
        required=False,
        help="The sub model path of the `pretrained_model_name_or_path`",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="whether or not to use LoRA.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--rl_prompt_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Number of epochs to train for. Each epoch is one round of sampling from the model "
        "followed by training on those samples.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_lcm", action="store_true", help="Whether or not to use LCM."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_inner_epochs",
        type=int,
        default=1,
        help="Number of inner epochs per outer epoch. Each inner epoch is one iteration "
        "through the data collected during one outer epoch's round of sampling.",
    )
    parser.add_argument(
        "--cfg",
        action="store_true",
        help="Whether or not to use classifier-free guidance during training. if enabled, the same guidance "
        "scale used during sampling will be used during training.",
    )
    parser.add_argument(
        "--adv_clip_max",
        type=float,
        default=5.0,
        help="Clip advantages to the range [-train_adv_clip_max, train_adv_clip_max].",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=1e-4,
        help="the PPO clip range.",
    )

    # Prompt Function and Reward Function
    parser.add_argument(
        "--prompt_fn",
        type=str,
        default="facechain",
        help="The prompt function to use.",
    )
    parser.add_argument(
        "--reward_fn",
        type=str,
        default="face_similarity",
        help="The reward function to use.",
    )
    parser.add_argument(
        "--target_image_dir",
        type=str,
        required=True,
        help="target_image_dir.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default) and `"wandb"`.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    if args.resume_from:
        args.resume_from = os.path.normpath(os.path.expanduser(args.resume_from))
        if "checkpoint_" not in os.path.basename(args.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(args.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {args.resume_from}")
            args.resume_from = os.path.join(
                args.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    accelerator_config = ProjectConfiguration(
        project_dir=args.logdir,
        automatic_checkpoint_naming=True,
        # total_limit=args.num_checkpoint_limit,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    accelerator = Accelerator(
        log_with=args.report_to,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want args.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.logdir is not None:
            os.makedirs(args.logdir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("\n".join(f'{k}: {v}' for k, v in vars(args).items()))

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(args.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    args.pretrained_model_name_or_path = snapshot_download(
                    args.pretrained_model_name_or_path, revision=args.revision)
    if args.sub_path is not None and len(args.sub_path) > 0:
        args.pretrained_model_name_or_path = os.path.join(args.pretrained_model_name_or_path, args.sub_path)

    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if args.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.rank
            )
        unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(unet.attn_processors)
    else:
        trainable_layers = unet
    pipeline = StableDiffusionPipeline(
        tokenizer=tokenizer,
        scheduler=scheduler,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        safety_checker=None,
        feature_extractor=None,
    )
    # merge_lora(pipeline, args.face_lora_path, 1.)
    

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)


    if args.use_lcm:
        try:
            from diffusers import LCMScheduler
        except:
            raise ImportError('diffusers version is not right, please update diffsers to >=0.22')
        lcm_model_path = snapshot_download('eavesy/lcm-lora-sdv1-5')
        pipeline.load_lora_weights(lcm_model_path)
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        args.sample_num_steps = 8
        args.sample_guidance_scale = 2
    # else:
    #     pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config)
    #     pipeline_ori.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline_ori.scheduler.config)
    #     args.sample_num_steps = 40
    #     args.sample_guidance_scale = 7
    pipeline.scheduler.set_timesteps(args.sample_num_steps)
    
    pipeline = merge_lora(pipeline, args.face_lora_path, 0.95, from_safetensor=args.face_lora_path.endswith('safetensors'))
    


    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if args.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    pipeline_ori = copy.deepcopy(pipeline)

    kl_ctl = AdaptiveKLController(0.2, 1, 1000)

    # set up diffusers-friendly checkpoint saving with Accelerate

    reward_mean_list, reward_std_list = [], []
    cur_best_reward_mean, reward_mean_heap = (float("-inf"), ""), []

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if args.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not args.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if args.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                args.pretrained.model, revision=args.pretrained.revision, subfolder="unet", use_safetensors=False
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not args.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet", use_safetensors=False)
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )










    # prepare prompt and reward fn
    if hasattr(reward_optimization.prompts, args.prompt_fn):
        prompt_fn = getattr(reward_optimization.prompts, args.prompt_fn)
    else:
        raise ValueError(
            "Prompt function {} is not defined in {}/reward_optimization/prompts.py."
            "".format(args.prompt_fn, os.path.abspath(__file__))
        )
    prompt_labeled_dir = args.target_image_dir + '_labeled'

    if hasattr(reward_optimization.rewards, args.reward_fn):
        
        reward_fn = getattr(reward_optimization.rewards, args.reward_fn)(target_image_dir=args.target_image_dir,
            device=accelerator.device,accelerator=accelerator,
            torch_dtype=inference_dtype)
    else:
        raise ValueError(
            "Reward function {} is not defined in {}/reward_optimization/rewards.py"
            "".format(args.reward_fn, os.path.abspath(__file__))
        )

    # generate negative prompt embeddings
    neg_prompt = "(nsfw:2), paintings, sketches, (worst quality:2), (low quality:2), lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, bad hand, tattoo, (username, watermark, signature, time signature, timestamp, artist name, copyright name, copyright),low res, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, strange fingers, bad hand, mole, ((extra legs)), ((extra hands))"
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [neg_prompt],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(args.sample_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(args.train_batch_size, 1, 1)


    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if args.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    # Train!
    samples_per_epoch = args.sample_batch_size * accelerator.num_processes * args.sample_num_batches_per_epoch
    total_train_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num Epochs = {args.num_epochs}")
    print(f"  Sample batch size per device = {args.sample_batch_size}")
    print(f"  Train batch size per device = {args.train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print("")
    print(f"  Total number of samples per epoch = {samples_per_epoch}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    # print(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    print(f"  Number of inner epochs = {args.num_inner_epochs}")

    assert args.sample_batch_size >= args.train_batch_size
    assert args.sample_batch_size % args.train_batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        accelerator.load_state(args.resume_from)
        first_epoch = int(args.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0
    
    user_id = os.path.basename(os.path.dirname(args.logdir))
    # check log path
    if accelerator.is_main_process:
        output_log = open(args.cache_log_file, 'w')


    global_step = 0
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    pipeline_ori.scheduler.alphas_cumprod = pipeline_ori.scheduler.alphas_cumprod.to(accelerator.device)
    
    prompts_abs_idx = 0
    prompts_eval_abs_idx = 0
    prompts_names = [args.rl_prompt_name]
    for epoch in range(first_epoch, args.num_epochs):
        
        if epoch%10 == 0:

            print("***** Running evaluating *****")
            pipeline.unet.eval()
            samples = []
            prompts = []
            # generate prompts
            prompts_and_style_path = [prompt_fn(prompts_names[_%len(prompts_names)], processed_dir=prompt_labeled_dir) for _ in range(args.sample_batch_size)]
            
            prompts = [each[0] for each in prompts_and_style_path]
            style_path = [each[1] for each in prompts_and_style_path]

            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            # sample
            with torch.no_grad():
                images, _, latents, log_probs = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=args.sample_num_steps,
                    guidance_scale=args.sample_guidance_scale,
                    eta=0.,
                    output_type="pt",
                )


            output_img_path = os.path.join(args.logdir, 'output_imgs_eval/')
            if not os.path.exists(output_img_path):
                os.makedirs(output_img_path, exist_ok=True)
            for i, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(output_img_path, f"{i}.jpg"))
            rewards = reward_fn(images, prompts)

            reward_mean_list.append(rewards.mean().cpu().detach().numpy())
            reward_std_list.append(rewards.std().cpu().detach().numpy())   

        #################### TRAINING ####################
        for inner_epoch in range(args.num_inner_epochs):

            # train
            pipeline.unet.train()
            info = defaultdict(list)

            latent = torch.randn((args.train_batch_size, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)    
            latent_ori = latent.clone()  
            # choose prompts where the index is from 0 to 4
            prompts_and_style_path = [prompt_fn(prompts_names[(prompts_abs_idx+_)%len(prompts_names)], processed_dir=prompt_labeled_dir) for _ in range(args.train_batch_size)]
            prompts_abs_idx = prompts_abs_idx + args.train_batch_size
            prompts = [each[0] for each in prompts_and_style_path]
            style_path = [each[1] for each in prompts_and_style_path]
            prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)   
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]   
            pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
            
            if args.use_lcm and isinstance(pipeline.scheduler, LCMScheduler):
                timesteps = torch.cat((pipeline.scheduler.timesteps,torch.tensor([0]).to(accelerator.device, dtype=torch.long))).to(accelerator.device)
            else:
                timesteps = pipeline.scheduler.timesteps

            ts_rewards = []
            with accelerator.accumulate(unet):
                    with autocast():
                        with torch.enable_grad(): # important b/c don't have on by default in module                        
                            keep_input = True
                            
                            ims, latents, log_probs = pipeline_with_logprob_with_grad(
                                pipeline,
                                prompt_embeds=prompt_embeds,
                                negative_prompt_embeds=train_neg_prompt_embeds,
                                num_inference_steps=args.sample_num_steps,
                                guidance_scale=args.sample_guidance_scale,
                                eta=1.,
                                output_type="pt",
                            )

                            _, _, latents_ori, log_probs_ori = pipeline_with_logprob(
                                pipeline_ori,
                                prompt_embeds=prompt_embeds,
                                negative_prompt_embeds=train_neg_prompt_embeds,
                                num_inference_steps=args.sample_num_steps,
                                guidance_scale=args.sample_guidance_scale,
                                eta=1.,
                                output_type="pt",
                            )
                            ts_rewards = (torch.stack(log_probs, dim=0) - torch.stack(log_probs_ori, dim=0))

                            image_rewards = reward_fn(ims, prompts, train=True)
                            mean_kl = ts_rewards.sum(dim=0).mean()
                            kl_ctl.update(mean_kl.detach().cpu().numpy(), args.train_batch_size)
                            KL_coef = torch.tensor(kl_ctl.value).to(ts_rewards.device, dtype=ts_rewards.dtype)
                            
                            ts_rewards = ts_rewards * KL_coef
                            
                            ts_rewards[-1] += image_rewards
                            # final_rewards = ts_rewards
                            final_rewards = ts_rewards.sum(dim=0)
                            loss = - final_rewards.mean()

                            loss = loss * 0.1
                            
                            print(loss.item(), image_rewards.mean().item())
                        
                            info["loss"].append(loss)
                            # backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(trainable_layers.parameters(), args.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()


                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        print(f'info: {info}')
                        # print(info)
                        # accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

        # make sure we did an optimization step at the end of the inner epoch
        # assert accelerator.sync_gradients

        # Save the best checkpoint and maintain a heap (except for epoch 0) with length `num_checkpoint_limit - 1`
        # by `reward_mean`. Note that, `reward_mean` corresponds to the model saved in last epoch.
        if epoch % args.save_freq == 0 and accelerator.is_main_process:
            if reward_mean_list[-1] >= cur_best_reward_mean[0]:
                # (reward_mean, -1) => baseline will not be copied to best_outputs.
                cur_best_reward_mean = (reward_mean_list[-1], accelerator.save_iteration - 1)
                best_ckpt_src_dir = os.path.join(
                    args.logdir, "checkpoints", "checkpoint_{}".format(accelerator.save_iteration - 1)
                )
                if os.path.exists(best_ckpt_src_dir):
                    best_ckpt_dst_dir = os.path.join(args.logdir, "best_outputs")
                    if os.path.exists(best_ckpt_dst_dir):
                        shutil.rmtree(best_ckpt_dst_dir)
                    shutil.copytree(best_ckpt_src_dir, best_ckpt_dst_dir)
                    print(
                        "Copy the checkpoint directory: {} with the highest reward {} to best_outputs"
                        .format(best_ckpt_src_dir, cur_best_reward_mean)
                    )
            if len(reward_mean_heap) < args.num_checkpoint_limit - 1:
                if accelerator.save_iteration > 0:
                    heapq.heappush(reward_mean_heap, (reward_mean_list[-1], accelerator.save_iteration - 1))
            else:
                reward_save_iteration = (reward_mean_list[-1], accelerator.save_iteration - 1)
                if reward_mean_list[-1] >= reward_mean_heap[0][0]:
                    reward, save_iteration = heapq.heappushpop(reward_mean_heap, reward_save_iteration)
                    popped_ckpt_dir = os.path.join(
                        args.logdir, "checkpoints", "checkpoint_{}".format(save_iteration)
                    )
                    if os.path.exists(popped_ckpt_dir):
                        shutil.rmtree(popped_ckpt_dir)
                        print(
                            "Delete the checkpoint directory: {} with the smallest reward {} in the heap"
                            .format(popped_ckpt_dir, reward)
                        )
                else:
                    last_ckpt_dir = os.path.join(
                        args.logdir, "checkpoints", "checkpoint_{}".format(accelerator.save_iteration - 1)
                    )
                    shutil.rmtree(last_ckpt_dir)
                    print(
                        "Delete last checkpoint directory: {} with smaller reward {}"
                        .format(last_ckpt_dir, reward_mean_list[-1])
                    )
            accelerator.save_state()
            np.savetxt(os.path.join(args.logdir, "reward_mean.txt"), np.array(reward_mean_list), delimiter=",", fmt="%.4f")
            np.savetxt(os.path.join(args.logdir, "reward_std.txt"), np.array(reward_std_list), delimiter=",", fmt="%.4f")
    
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # accelerator.save_model(trainable_layers, safetensor_save_path, safe_serialization=True)
        # we will remove cache_log_file after train
        with open(args.cache_log_file, "w") as _:
            pass

if __name__ == "__main__":
    main()
