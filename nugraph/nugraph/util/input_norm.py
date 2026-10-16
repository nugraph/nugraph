"""Input feature normalization module"""
import torch
import torch.distributed as dist

class BufferDict(torch.nn.Module):
    """
    Dictionary of buffers

    Running statistics are buffers, so neither the optimizer nor DDP treats
    them as parameters, but they keep the state_dict keys they had when they
    were held in a ParameterDict.
    """
    def __init__(self, buffers: dict[str, torch.Tensor]):
        super().__init__()
        for key, value in buffers.items():
            self.register_buffer(key, value)

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)

class InputNorm(torch.nn.Module):
    """
    PyTorch module to normalize input features

    Args:
        num_feats: Number of tensor features
    """
    __jit_unused_properties__ = ["update"]

    def __init__(self, num_features: int):
        super().__init__()

        # hold onto the running averages for the mean and variance. the
        # variance starts at one, so the module is the identity until it
        # has seen data (eg. in the sanity check before training)
        self.norm = BufferDict({
            "mean": torch.zeros(num_features),
            "var": torch.ones(num_features),
            "count": torch.zeros(1, dtype=torch.long)})

        # whether the running averages are frozen, saved in checkpoints
        self.register_buffer("frozen", torch.tensor(False))
        self._update = True

    @property
    def update(self) -> bool:
        """Whether to continue updating running averages"""
        return self._update

    @update.setter
    def update(self, value: bool) -> None:
        self._update = bool(value)
        self.frozen.fill_(not value)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints from before the flag was saved: the statistics were
        # frozen at the end of the first training epoch, before the
        # end-of-epoch checkpoint was written, so any that hold data are frozen
        count = state_dict.get(prefix + "norm.count")
        if prefix + "frozen" not in state_dict and count is not None:
            state_dict[prefix + "frozen"] = (count > 0).squeeze()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._update = not bool(self.frozen)

    @staticmethod
    def upgrade_checkpoint(model: torch.nn.Module, checkpoint: dict) -> None:
        """
        Drop InputNorm statistics from an old checkpoint's optimizer state

        The statistics used to be parameters, so the optimizer state saved
        by older code counts them. Removing them lets those checkpoints
        resume training.

        Args:
            model: Model the checkpoint is loaded into
            checkpoint: Lightning checkpoint dictionary
        """
        params = {name for name, _ in model.named_parameters()}
        stats = {f"{name}.norm.{key}" for name, module in model.named_modules()
                 if isinstance(module, InputNorm) for key in ("mean", "var", "count")}
        saved = [key for key in checkpoint["state_dict"] if key in params or key in stats]
        if len(saved) == len(params):
            return
        keep = [i for i, key in enumerate(saved) if key not in stats]
        remap = {old: new for new, old in enumerate(keep)}
        for optimizer in checkpoint.get("optimizer_states", []):
            for group in optimizer["param_groups"]:
                group["params"] = [remap[i] for i in group["params"] if i in remap]
            optimizer["state"] = {remap[i]: v for i, v in optimizer["state"].items() if i in remap}

    @torch.jit.unused
    def batch_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Count, mean and variance of a batch, combined across GPUs

        Args:
            x: Tensor to compute statistics for
        """
        # sum in float64, which also avoids bf16 precision loss
        x = x.double()
        f = x.size(1)
        s = torch.cat((x.new_full((1,), x.size(0)), x.sum(dim=0), x.square().sum(dim=0)))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(s, op=dist.ReduceOp.SUM)
        n, s1, s2 = s[0], s[1:f+1], s[f+1:]
        mean = s1 / n
        var = (s2 - n * mean.square()) / (n - 1)
        return n.long(), mean.float(), var.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor: # pylint: disable=arguments-differ
        """
        Forward pass for InputNorm module

        Args:
            x: Tensor to normalize
        """

        # update running average
        if self._update and self.training:

            n1, m1, v1 = self.norm.count, self.norm.mean, self.norm.var
            n2, m2, v2 = self.batch_stats(x)

            n = n1 + n2
            mean = ((n1*m1) + (n2*m2)) / n
            d1 = v1 + (mean - m1).square()
            d2 = v2 + (mean - m2).square()
            var = ((n1*d1) + (n2*d2)) / n

            self.norm.count.copy_(n)
            self.norm.mean.copy_(mean)
            self.norm.var.copy_(var)


        # return normalized tensor
        return (x - self.norm.mean[None, :]) / (self.norm.var[None, :] + 1e-5).sqrt()
