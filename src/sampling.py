import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from misc import set_grad
from tqdm import tqdm
import os


class Sampler:
    def __init__(
        self,
        session_dir,
        Gen,
        Cls,
        d_cond_mtd,
        dataset_name,
        z_dim,
        gammas,
        temperature_logits,
        temperature_validity,
        device
    ):
        self.session_dir = session_dir
        self.Gen = Gen
        self.Cls = Cls
        self.d_cond_mtd = d_cond_mtd
        self.z_dim = z_dim
        self.gammas = torch.tensor(gammas, dtype=torch.float32, device=device).view(1, -1)
        self.temperature_logits = temperature_logits
        self.temperature_validity = temperature_validity
        self.device = device
        
        if dataset_name == 'cifar7to3':
            self.padding=1
        else:
            self.padding=2

        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
    
    def get_samples(self, batch_size, label):
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        fake_images = self.Gen(z, label)
        return fake_images
    
    def compute_target_ratio(self, x, intersection_indices, difference_indices, label):
        real_logits, fake_logits, validity = self.Cls(x)
        prob_real_class = F.softmax(real_logits/self.temperature_logits, dim=-1)
        prob_real = torch.sigmoid(validity/self.temperature_validity).view(-1, 1)

        weighted_class_ratio = self.gammas * prob_real_class

        if difference_indices:
            diff = torch.min(weighted_class_ratio[:, intersection_indices], dim=1, keepdim=True)[0] \
            - torch.max(weighted_class_ratio[:, difference_indices], dim=1, keepdim=True)[0]
        else:
            diff = torch.min(weighted_class_ratio[:, intersection_indices], dim=1, keepdim=True)[0]
        target_ratio = torch.clamp(diff, min=0.0) * prob_real / (1.0 - prob_real)
        if self.d_cond_mtd == "PD":
            prob_fake_class =  F.softmax(fake_logits, dim=-1)
            target_ratio = target_ratio / torch.gather(prob_fake_class, 1, label.view(-1, 1))
        return target_ratio
    
    def mh(self, batch_size, intersection_indices, difference_indices, n_steps, save_img=False, save_every=10, eps=1e-5):
        if self.d_cond_mtd == "W/O":
            label = None
        elif self.d_cond_mtd == "PD":
            label = torch.tensor(intersection_indices[0], device=self.device).expand(batch_size)
        else:
            raise NotImplementedError
        
        x = self.get_samples(batch_size, label)
        
        if save_img:
            save_image((x+1)/2, os.path.join(self.session_dir, f'before.png'), pad_value=1.0, padding=self.padding)

        for step in tqdm(range(n_steps)):
            y = self.get_samples(batch_size, label)
            accept_prob = torch.clamp(self.compute_target_ratio(y, intersection_indices, difference_indices, label) / (self.compute_target_ratio(x, intersection_indices, difference_indices, label) + eps), max=1.0)
            
            mask = torch.bernoulli(accept_prob)
            mask_x = mask.view(-1, 1, 1, 1)
            x = mask_x * y + (1.0 - mask_x) * x

            if save_img and (step+1) % save_every == 0:
                save_image((x+1)/2, os.path.join(self.session_dir, f'step_{step+1}.png'), pad_value=1.0, padding=self.padding)
        
        if save_img:
            save_image((x+1)/2, os.path.join(self.session_dir, 'after.png'), pad_value=1.0, padding=self.padding)
