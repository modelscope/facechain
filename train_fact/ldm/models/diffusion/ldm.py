import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from ldm.util import default
from ldm.modules.diffusionmodules.util import  extract_into_tensor
from .ddpm import DDPM



class LatentDiffusion(DDPM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # hardcoded 
        self.clip_denoised = False
        


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    "Does not support DDPM sampling anymore. Only do DDIM or PLMS"

    # = = = = = = = = = = = = Below is for sampling = = = = = = = = = = = = # 

    # def predict_start_from_noise(self, x_t, t, noise):
    #     return ( extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
    #              extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise )

    # def q_posterior(self, x_start, x_t, t):
    #     posterior_mean = (
    #             extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
    #             extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
    #     )
    #     posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
    #     posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
    #     return posterior_mean, posterior_variance, posterior_log_variance_clipped


    # def p_mean_variance(self, model, x, c, t):

    #     model_out = model(x, t, c)
    #     x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

    #     if self.clip_denoised:
    #         x_recon.clamp_(-1., 1.)

    #     model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
    #     return model_mean, posterior_variance, posterior_log_variance, x_recon


    # @torch.no_grad()
    # def p_sample(self, model, x, c, t):
    #     b, *_, device = *x.shape, x.device
    #     model_mean, _, model_log_variance, x0 = self.p_mean_variance(model, x=x, c=c, t=t, )
    #     noise = torch.randn_like(x) 

    #     # no noise when t == 0
    #     nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

    #     return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0


    # @torch.no_grad()
    # def p_sample_loop(self, model, shape, c):
    #     device = self.betas.device
    #     b = shape[0]
    #     img = torch.randn(shape, device=device)

    #     iterator = tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps) 
    #     for i in iterator:
    #         ts = torch.full((b,), i, device=device, dtype=torch.long)
    #         img, x0 = self.p_sample(model, img, c, ts)

    #     return img


    # @torch.no_grad()
    # def sample(self, model, shape, c, uc=None, guidance_scale=None):
    #     return self.p_sample_loop(model, shape, c)





