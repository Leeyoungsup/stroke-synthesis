import torch
from tqdm import tqdm

# --- EDM2 Noise Schedule Class ---
class EDM2Schedule:
    def __init__(self, steps, P_mean=-1.2, P_std=1.2, rho=7.0, device='cuda'):
        self.steps = steps
        self.P_mean = P_mean
        self.P_std = P_std
        self.rho = rho
        self.device = device
        self.sigmas = self._generate_schedule()

    def _generate_schedule(self):
        sigma_max = torch.exp(torch.tensor(self.P_mean + 2 * self.P_std))
        sigma_min = torch.exp(torch.tensor(self.P_mean - 2 * self.P_std))

        t = torch.linspace(0, 1, self.steps, device=self.device)
        inv_rho = 1.0 / self.rho
        sigmas = (sigma_max ** inv_rho + t * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** self.rho
        return sigmas

# --- EDM2 Sampler Class ---
class EDM2Sampler:
    def __init__(self, model, schedule, sampler_type='heun', cfg_scale=2.0, device='cuda'):
        self.model = model
        self.schedule = schedule
        self.sampler_type = sampler_type.lower()
        self.cfg_scale = cfg_scale
        self.device = device

    @torch.no_grad()
    def sample(self, shape, class_labels=None):
        x = torch.randn(shape, device=self.device) * self.schedule.sigmas[0]

        for i in tqdm(range(len(self.schedule.sigmas) - 1), desc=f'Sampling ({self.sampler_type})'):
            sigma_curr = self.schedule.sigmas[i]
            sigma_next = self.schedule.sigmas[i + 1]
            x = self._step(x, sigma_curr, sigma_next, class_labels)

        return x

    def _step(self, x, sigma_curr, sigma_next, class_labels):
        if self.sampler_type == 'euler':
            return self._euler_step(x, sigma_curr, sigma_next, class_labels)
        elif self.sampler_type == 'heun':
            return self._heun_step(x, sigma_curr, sigma_next, class_labels)
        elif self.sampler_type == 'dpm++':
            return self._dpmpp_step(x, sigma_curr, sigma_next, class_labels)
        else:
            raise ValueError(f"Unsupported sampler type: {self.sampler_type}")

    def _euler_step(self, x, sigma_curr, sigma_next, class_labels):
        sigma_tensor = torch.full((x.size(0),), sigma_curr, device=self.device)
        denoised = self._denoise(x, sigma_tensor, class_labels)
        d = (x - denoised) / sigma_curr
        return x + (sigma_next - sigma_curr) * d

    def _heun_step(self, x, sigma_curr, sigma_next, class_labels):
        sigma_tensor = torch.full((x.size(0),), sigma_curr, device=self.device)
        denoised = self._denoise(x, sigma_tensor, class_labels)
        d = (x - denoised) / sigma_curr
        x_euler = x + (sigma_next - sigma_curr) * d

        sigma_next_tensor = torch.full((x.size(0),), sigma_next, device=self.device)
        denoised_next = self._denoise(x_euler, sigma_next_tensor, class_labels)
        d_prime = (x_euler - denoised_next) / sigma_next
        return x + 0.5 * (sigma_next - sigma_curr) * (d + d_prime)

    def _dpmpp_step(self, x, sigma_curr, sigma_next, class_labels):
        # Simple DPM++-like step (approximate version)
        sigma_tensor = torch.full((x.size(0),), sigma_curr, device=self.device)
        denoised = self._denoise(x, sigma_tensor, class_labels)
        d = (x - denoised) / sigma_curr
        return x + (sigma_next - sigma_curr) * d

    def _denoise(self, x, sigma_tensor, class_labels):
        denoised_cond = self.model(x, sigma_tensor, class_labels)
        if class_labels is not None:
            denoised_uncond = self.model(x, sigma_tensor, None)
            denoised = denoised_uncond + self.cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = denoised_cond
        return denoised
