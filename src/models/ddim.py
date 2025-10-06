
import torch
# ... (UNet and other imports from ddpm.py)

class DDIM:
    def __init__(self, model, timesteps=1000, eta=0.0): # eta=0 for deterministic DDIM
        self.model = model
        self.timesteps = timesteps
        self.eta = eta
        # Reuse beta/alpha schedules from DDPM for consistency or generate new ones
        self.ddpm = DDPM(model, timesteps=timesteps) # Use DDPM's schedule prep

    @torch.no_grad()
    def sample_ddim(self, shape, ddim_steps, guidance_scale=1.0, verbose_steps=10):
        # DDIM specific sampling steps (subsampling timesteps)
        ddim_timestep_seq = torch.linspace(0, self.timesteps - 1, ddim_steps).long()
        alphas_cumprod = self.ddpm.alphas_cumprod
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        img = torch.randn(shape)
        intermediate_images = []

        for i, timestep in enumerate(reversed(ddim_timestep_seq)):
            t = torch.full((1,), timestep, dtype=torch.long)
            # Predict noise using the UNet
            predicted_noise = self.model(img, t)

            # DDIM sampling equation
            alpha_cumprod_t = alphas_cumprod[t]
            alpha_cumprod_t_prev = alphas_cumprod_prev[t]

            # x_0_pred calculation (equation from DDIM paper)
            pred_x0 = (img - torch.sqrt(1. - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

            # Variance calculation (eta determines stochasticity)
            sigma_t = self.eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * \
                      torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1. - alpha_cumprod_t_prev - sigma_t**2) * predicted_noise

            if i < ddim_steps - 1:
                noise = torch.randn_like(img)
                img = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma_t * noise
            else:
                img = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt # No noise for final step

            if (ddim_steps - 1 - i) % verbose_steps == 0 or i == ddim_steps - 1:
                intermediate_images.append(img.cpu().squeeze().numpy())

        return img, intermediate_images
