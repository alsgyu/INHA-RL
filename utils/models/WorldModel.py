import torch


class WorldModel(torch.nn.Module):
    """Small one-step dynamics model for auxiliary and off-policy training."""

    def __init__(self, obs_dim, action_dim, hidden_dims=None, command_dim=0, velocity_dim=0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.command_dim = int(command_dim)
        self.velocity_dim = int(velocity_dim)
        hidden_dims = hidden_dims or [512, 512]

        input_dim = self.obs_dim + self.action_dim + self.command_dim
        output_dim = self.obs_dim + 1 + 1 + self.velocity_dim
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(last_dim, int(hidden_dim)))
            layers.append(torch.nn.ELU())
            last_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(last_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, obs, action, command=None):
        inputs = [obs, action]
        if self.command_dim > 0:
            if command is None:
                command = torch.zeros(*obs.shape[:-1], self.command_dim, dtype=obs.dtype, device=obs.device)
            inputs.append(command)
        output = self.net(torch.cat(inputs, dim=-1))
        delta_obs = output[..., : self.obs_dim]
        reward = output[..., self.obs_dim]
        done_logit = output[..., self.obs_dim + 1]
        result = {
            "delta_obs": delta_obs,
            "reward": reward,
            "done_logit": done_logit,
        }
        if self.velocity_dim > 0:
            start = self.obs_dim + 2
            result["velocity"] = output[..., start : start + self.velocity_dim]
        return result


class QNetwork(torch.nn.Module):
    """Twin-Q building block for off-policy continuous control."""

    def __init__(self, obs_dim, action_dim, privileged_obs_dim=0, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [512, 512]
        input_dim = int(obs_dim) + int(action_dim) + int(privileged_obs_dim)
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(last_dim, int(hidden_dim)))
            layers.append(torch.nn.ELU())
            last_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(last_dim, 1))
        self.net = torch.nn.Sequential(*layers)
        self.privileged_obs_dim = int(privileged_obs_dim)

    def forward(self, obs, action, privileged_obs=None):
        inputs = [obs, action]
        if self.privileged_obs_dim > 0:
            if privileged_obs is None:
                privileged_obs = torch.zeros(*obs.shape[:-1], self.privileged_obs_dim, dtype=obs.dtype, device=obs.device)
            inputs.append(privileged_obs)
        return self.net(torch.cat(inputs, dim=-1)).squeeze(-1)
