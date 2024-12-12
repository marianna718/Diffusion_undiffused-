import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator:torch.Generator, num_training_steps=1000, beta_start:float= 0.00085, beta_end:float = 0.0120):
        # the parametrs used for noise reduction and maybe detection as well are the beta and alpha
        # betta is the variance of the noise distribution, and beta_start and end are defining that distribution
        self.betas = torch.linspace(beta_start ** 0.5, beta_end**0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps = 50):
        self.num_inference_steps = num_inference_steps
        # 999,999,997,996, .....
        # 999, 999-20, 999-40, 999-60, ... each time less by 20
        # becouse 1000/ 50 = 20
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)


    def _get_previous_timestep(self, timestep:int) ->int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)

        return prev_t


    def _get_variance(self, timestep:int)-> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_pred_t = self.alpha_cumprod[timestep]
        alpha_pred_t_prev = 1 - self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_pred_t / alpha_pred_t_prev
        # by the formaula form paper DDPM (7)
        variance = (1-alpha_pred_t_prev) / (1- alpha_pred_t) * current_beta_t
        variance = torch.clamp(variance, min = 1e-20)

        return variance

    def set_strength(self, strength = 1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step]
        self.start_step = start_step




    # given the timestep at whiche the noise was added or we think was added
    # at that timestep we are telling to remove the noise

    # the output would be the predicted epsilon tetha see in the paper
    def step(self, timestep:int, latents: torch.Tensor, model_output:torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one  
        # if we dont have the prvious step we are returnint 1, if we do then alpha
        beta_prod_t = 1 - alpha_prod_t_prev
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Compute the prediction original samples using formula (15) of the DDPM paper
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # compute the coefficients for pred_original_sample and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        # for calculating the variance we will use a function

        # becouse we need to add variance only if we are on the last timestep
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device= device, dtype= model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        # N(0,1) --> N(mu, sigma^2)
        # X = mu + sigma * Z where Z ~ N(0,1)
        pred_prev_sample = pred_prev_sample + variance

        return variance




    def add_noise(self, original_samples: torch.FloatTensor, timesteps:torch.IntTensor) -> torch.FloatTensor:
        alpha_cumpred = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        # timesteps is acctually a timestep
        sqrt_alpha_pred = alpha_cumpred[timesteps] ** 0.5
        sqrt_alpha_pred = sqrt_alpha_pred.flatten()
        while len(sqrt_alpha_pred.shape) < len(original_samples.shape):
            sqrt_alpha_pred = sqrt_alpha_pred.unsqueeze(-1)

        sqrt_one_minus_alpha_pred = (1 - alpha_cumpred[timesteps]) ** 0.5 # standard deviation
        sqrt_one_minus_alpha_pred = sqrt_one_minus_alpha_pred.flatten()
        while len(sqrt_one_minus_alpha_pred.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_pred = sqrt_one_minus_alpha_pred.unsqueeze(-1)

        # Accordint to the equation (4) of the DDPM paper
        # Z = N(0,1) -> N(mean, variance ) =X?
        # X = mean + stdev * Z
        noise = torch.randn(original_samples.shape, generator= self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_pred * original_samples) + (sqrt_one_minus_alpha_pred) * noise
        return noisy_samples
    
