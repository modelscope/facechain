# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# with the following modifications:
# - It uses the patched version of `ddim_step_with_logprob` from `ddim_with_logprob.py`. As such, it only supports the
#   `ddim` scheduler.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/reward_optimization/diffusers_patch/pipeline_with_logprob.py.

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch.utils.checkpoint as checkpoint
import torch
import random
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
from .ddim_with_logprob import ddim_step_with_logprob
from diffusers import LCMScheduler
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_lcm import LCMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
import math


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

def step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    with_logprob: bool = False,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
) -> Union[LCMSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
    Returns:
        [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.
    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    # 1. get previous step value
    prev_step_index = self.step_index + 1
    if prev_step_index < len(self.timesteps):
        prev_timestep = self.timesteps[prev_step_index]
    else:
        prev_timestep = timestep

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.to(sample.device))
    alpha_prod_t_prev = torch.where(
        prev_timestep.to(sample.device) >= 0, self.alphas_cumprod.gather(0, prev_timestep.to(sample.device)), self.final_alpha_cumprod
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # 3. Get scalings for boundary conditions
    c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
    c_skip = _left_broadcast(c_skip, sample.shape).to(sample.device)
    c_out = _left_broadcast(c_out, sample.shape).to(sample.device)

    # 4. Compute the predicted original sample x_0 based on the model parameterization
    if self.config.prediction_type == "epsilon":  # noise-prediction
        predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif self.config.prediction_type == "sample":  # x-prediction
        predicted_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":  # v-prediction
        predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction` for `LCMScheduler`."
        )

    # 5. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        predicted_original_sample = self._threshold_sample(predicted_original_sample)
    elif self.config.clip_sample:
        predicted_original_sample = predicted_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 6. Denoise model output using boundary conditions
    denoised = c_out * predicted_original_sample + c_skip * sample

    # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
    # Noise is not used on the final timestep of the timestep schedule.
    # This also means that noise is not used for one-step sampling.
    if self.step_index != self.num_inference_steps - 1:
        noise = randn_tensor(
            model_output.shape, generator=generator, device=model_output.device, dtype=denoised.dtype
        )
        prev_sample_mean = alpha_prod_t_prev.sqrt() * denoised
        prev_sample = prev_sample_mean + beta_prod_t_prev.sqrt() * noise
    else:
        prev_sample_mean = denoised
        prev_sample = denoised

    # upon completion increase step index by one
    self._step_index += 1
    

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((beta_prod_t_prev.sqrt()+1e-6)**2))
        - torch.log(beta_prod_t_prev.sqrt()+1e-6)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    
    prev_sample = prev_sample.type(sample.dtype)

    if with_logprob:
        if not return_dict:
            return (prev_sample, denoised), log_prob

        return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised), log_prob
    else:
        if not return_dict:
            return (prev_sample, denoised)
        return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)


@torch.no_grad()
def pipeline_with_logprob(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        if isinstance(self.scheduler, LCMScheduler):
            timesteps = torch.cat((self.scheduler.timesteps,torch.tensor([0]).to(device, dtype=torch.long))).to(device)
    
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LCMScheduler):
                LCMScheduler.step = step
                ret_dict, log_prob = self.scheduler.step(noise_pred, t.long(), latents, with_logprob=True)
                latents = ret_dict.prev_sample
            else:
                latents, log_prob = ddim_step_with_logprob(self.scheduler, noise_pred, t, latents, **extra_step_kwargs)

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return image, has_nsfw_concept, all_latents, all_log_probs



# def pipeline_with_logprob_with_grad(
#     self: StableDiffusionPipeline,
#     prompt: Union[str, List[str]] = None,
#     height: Optional[int] = None,
#     width: Optional[int] = None,
#     num_inference_steps: int = 50,
#     guidance_scale: float = 7.5,
#     negative_prompt: Optional[Union[str, List[str]]] = None,
#     num_images_per_prompt: Optional[int] = 1,
#     eta: float = 0.0,
#     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#     latents: Optional[torch.FloatTensor] = None,
#     prompt_embeds: Optional[torch.FloatTensor] = None,
#     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#     output_type: Optional[str] = "pil",
#     return_dict: bool = True,
#     callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
#     callback_steps: int = 1,
#     cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#     guidance_rescale: float = 0.0,
# ):
#     r"""
#     Function invoked when calling the pipeline for generation.

#     Args:
#         prompt (`str` or `List[str]`, *optional*):
#             The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
#             instead.
#         height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
#             The height in pixels of the generated image.
#         width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
#             The width in pixels of the generated image.
#         num_inference_steps (`int`, *optional*, defaults to 50):
#             The number of denoising steps. More denoising steps usually lead to a higher quality image at the
#             expense of slower inference.
#         guidance_scale (`float`, *optional*, defaults to 7.5):
#             Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
#             `guidance_scale` is defined as `w` of equation 2. of [Imagen
#             Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
#             1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
#             usually at the expense of lower image quality.
#         negative_prompt (`str` or `List[str]`, *optional*):
#             The prompt or prompts not to guide the image generation. If not defined, one has to pass
#             `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
#             less than `1`).
#         num_images_per_prompt (`int`, *optional*, defaults to 1):
#             The number of images to generate per prompt.
#         eta (`float`, *optional*, defaults to 0.0):
#             Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
#             [`schedulers.DDIMScheduler`], will be ignored for others.
#         generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
#             One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
#             to make generation deterministic.
#         latents (`torch.FloatTensor`, *optional*):
#             Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
#             generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
#             tensor will ge generated by sampling using the supplied random `generator`.
#         prompt_embeds (`torch.FloatTensor`, *optional*):
#             Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
#             provided, text embeddings will be generated from `prompt` input argument.
#         negative_prompt_embeds (`torch.FloatTensor`, *optional*):
#             Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
#             weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
#             argument.
#         output_type (`str`, *optional*, defaults to `"pil"`):
#             The output format of the generate image. Choose between
#             [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
#         return_dict (`bool`, *optional*, defaults to `True`):
#             Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
#             plain tuple.
#         callback (`Callable`, *optional*):
#             A function that will be called every `callback_steps` steps during inference. The function will be
#             called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
#         callback_steps (`int`, *optional*, defaults to 1):
#             The frequency at which the `callback` function will be called. If not specified, the callback will be
#             called at every step.
#         cross_attention_kwargs (`dict`, *optional*):
#             A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
#             `self.processor` in
#             [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
#         guidance_rescale (`float`, *optional*, defaults to 0.7):
#             Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
#             Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
#             [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
#             Guidance rescale factor should fix overexposure when using zero terminal SNR.

#     Examples:

#     Returns:
#         [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
#         [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
#         When returning a tuple, the first element is a list with the generated images, and the second element is a
#         list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
#         (nsfw) content, according to the `safety_checker`.
#     """
#     # 0. Default height and width to unet
#     height = height or self.unet.config.sample_size * self.vae_scale_factor
#     width = width or self.unet.config.sample_size * self.vae_scale_factor

#     # 2. Define call parameters
#     if prompt is not None and isinstance(prompt, str):
#         batch_size = 1
#     elif prompt is not None and isinstance(prompt, list):
#         batch_size = len(prompt)
#     else:
#         batch_size = prompt_embeds.shape[0]

#     device = self._execution_device

#     # 3. Encode input prompt
#     text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    
#     # 4. Prepare timesteps
#     self.scheduler.set_timesteps(num_inference_steps, device=device)
#     timesteps = self.scheduler.timesteps

#     # 5. Prepare latent variables
#     num_channels_latents = self.unet.config.in_channels
#     latents = self.prepare_latents(
#         batch_size * num_images_per_prompt,
#         num_channels_latents,
#         height,
#         width,
#         prompt_embeds.dtype,
#         device,
#         generator,
#         latents,
#     )


#     # 7. Denoising loop
#     num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
#     all_latents = [latents]
#     all_log_probs = []
#     with self.progress_bar(total=num_inference_steps) as progress_bar:
#         if isinstance(self.scheduler, LCMScheduler):
#             timesteps = torch.cat((self.scheduler.timesteps,torch.tensor([0]).to(device, dtype=torch.long))).to(device)
        
#         train_batch_size = len(prompt_embeds)
#         for i, t in enumerate(timesteps):
#             # expand the latents if we are doing classifier free guidance
#             t = torch.tensor([t], device=latents.device)
#             t = t.repeat(train_batch_size)
#             noise_pred_uncond = checkpoint.checkpoint(self.unet, latents, t.long(), negative_prompt_embeds, use_reentrant=False).sample
#             noise_pred_cond = checkpoint.checkpoint(self.unet, latents, t.long(), prompt_embeds, use_reentrant=False).sample
            
#             timestep = random.randint(0, num_inference_steps)
#             if i < timestep:
#                 noise_pred_uncond = noise_pred_uncond.detach()
#                 noise_pred_cond = noise_pred_cond.detach()

#             grad = (noise_pred_cond - noise_pred_uncond)
#             noise_pred = noise_pred_uncond + guidance_rescale * grad  
            
#             # compute the previous noisy sample x_t -> x_t-1
#             if isinstance(self.scheduler, LCMScheduler):
#                 LCMScheduler.step = step
#                 ret_dict, log_prob = self.scheduler.step(noise_pred, t.long(), latents, with_logprob=True)
#                 latents = ret_dict.prev_sample
#             else:
#                 latents, log_prob = ddim_step_with_logprob(self.scheduler, noise_pred, t, latents)

#             all_latents.append(latents)
#             all_log_probs.append(log_prob)


#     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
#     return image, all_latents, all_log_probs


def pipeline_with_logprob_with_grad(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )
    
    
    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    if isinstance(self.scheduler, LCMScheduler):
        timesteps = torch.cat((self.scheduler.timesteps,torch.tensor([0]).to(device, dtype=torch.long))).to(device)
    
    for i, t in enumerate(timesteps):
        # predict the noise residual

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = checkpoint.checkpoint(self.unet,
            latent_model_input,
            t,prompt_embeds, use_reentrant=False
            # cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        timestep = random.randint(max(num_inference_steps-10,0), num_inference_steps)
        if i < timestep:
            noise_pred_uncond = noise_pred_uncond.detach()
            noise_pred_text = noise_pred_text.detach()

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1
        if isinstance(self.scheduler, LCMScheduler):
            LCMScheduler.step = step
            ret_dict, log_prob = self.scheduler.step(noise_pred, t.long(), latents, with_logprob=True)
            latents = ret_dict.prev_sample
        else:
            latents, log_prob = ddim_step_with_logprob(self.scheduler, noise_pred, t, latents, **extra_step_kwargs)

        all_latents.append(latents)
        all_log_probs.append(log_prob)

    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
    
    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=[True] * image.shape[0])
    return image, all_latents, all_log_probs