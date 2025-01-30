import torch

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)
        
    def add_noise(self, x_0, t, device):
        noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].to(device)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].to(device)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise
    
    def sample_timesteps(self, n, device):
        return torch.randint(low=1, high=self.timesteps, size=(n,), device=device)
