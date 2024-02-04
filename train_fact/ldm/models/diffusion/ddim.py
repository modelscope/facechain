import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, diffusion, model, schedule="linear", alpha_generator_func=None, set_alpha_scale=None):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.device = diffusion.betas.device
        self.ddpm_num_timesteps = diffusion.num_timesteps
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func
        self.set_alpha_scale = set_alpha_scale
        

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)


    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=False)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=False)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    @torch.no_grad()
    def sample(self, S, shape, input, uc=None, guidance_scale=1, mask=None, x0=None):
        self.make_schedule(ddim_num_steps=S)
        return self.ddim_sampling(shape, input, uc, guidance_scale,  mask=mask, x0=x0)
 

    @torch.no_grad()
    def ddim_sampling(self, shape, input, uc, guidance_scale=1, mask=None, x0=None):
        b = shape[0]
        
        img = input["x"]
        if img == None:     
            img = torch.randn(shape, device=self.device)
            input["x"] = img


        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]

        #iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        iterator = time_range
  
        if self.alpha_generator_func != None:
            alphas = self.alpha_generator_func(len(iterator))


        for i, step in enumerate(iterator):

            # set alpha 
            if self.alpha_generator_func != None:
                self.set_alpha_scale(self.model, alphas[i])
                if  alphas[i] == 0:
                    self.model.restore_first_conv_from_SD()
                    
            # run 
            index = total_steps - i - 1
            input["timesteps"] = torch.full((b,), step, device=self.device, dtype=torch.long)
            
            if mask is not None:
                assert x0 is not None
                img_orig = self.diffusion.q_sample( x0, input["timesteps"] ) 
                img = img_orig * mask + (1. - mask) * img
                input["x"] = img
            
            img, pred_x0 = self.p_sample_ddim(input, index=index, uc=uc, guidance_scale=guidance_scale)
            input["x"] = img 

        return img


    @torch.no_grad()
    def p_sample_ddim(self, input, index, uc=None, guidance_scale=1):


        e_t = self.model(input) 
        if uc is not None and guidance_scale != 1:
            unconditional_input = dict(x=input["x"], timesteps=input["timesteps"], context=uc, inpainting_extra_input=input["inpainting_extra_input"], grounding_extra_input=input['grounding_extra_input'])
            e_t_uncond = self.model( unconditional_input ) 
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)

        # select parameters corresponding to the currently considered timestep
        b = input["x"].shape[0] 
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)

        # current prediction for x_0
        pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like( input["x"] ) 
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0
