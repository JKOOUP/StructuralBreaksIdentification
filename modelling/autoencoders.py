import torch
import typing as tp
import torch.nn as nn


class TimeDomainAE(nn.Module):
    def __init__(self, num_features: int, num_time_invariant_features: int, num_instantaneous_features: int):
        super().__init__()
        self.num_time_inv_features: int = num_time_invariant_features
        self.num_inst_features: int = num_instantaneous_features
        self.hidden_dim: int = self.num_inst_features + self.num_time_inv_features

        self.encoder = nn.Sequential(
            nn.Linear(num_features, self.hidden_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, num_features),
            nn.Tanh(),
        )
    
    def forward(self, input: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        latent: torch.Tensor = self.encoder(input)
        time_inv_features: torch.Tensor = latent[:, :self.num_time_inv_features]
        inst_features: torch.Tensor = latent[:, -self.num_inst_features:]
        reconstruction: torch.Tensor = self.decoder(latent)
        return reconstruction, time_inv_features, inst_features


class FrequencyDomainAE(TimeDomainAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
