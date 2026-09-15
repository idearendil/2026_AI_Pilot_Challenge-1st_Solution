"""Frozen Junhwa actor adapter; assumes the shared 214D observation/action contract.

Checkpoint shapes establish architecture, not observation semantics. Inference is
Linear/Tanh body -> optional GRUCell -> four categorical action heads. Original
forward-source parity remains separate from structural checkpoint validation.
"""
import torch
from torch import nn


class JunhwaPolicy(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        cfg, weights = checkpoint['cfg'], checkpoint['model']
        if (cfg.get('activation', 'tanh') != 'tanh' or cfg.get('gru_all', False)
                or cfg.get('continuous_action', False)):
            raise ValueError('unsupported Junhwa actor configuration')
        self.num_bins = int(cfg.get('num_bins', 21))
        layers = []; width = 214; index = 0
        while f'actor_body.{index}.weight' in weights:
            w = weights[f'actor_body.{index}.weight']
            if w.ndim != 2 or w.shape[1] != width:
                raise ValueError('actor body requires compatible 214D input')
            layers.extend((nn.Linear(width, w.shape[0]), nn.Tanh()))
            width = w.shape[0]; index += 2
        if not layers:
            raise ValueError('missing actor body')
        self.actor_body = nn.Sequential(*layers)
        self.actor_gru = None
        if cfg.get('gru_last', False):
            self.actor_gru = nn.GRUCell(width, width)
        self.actor_logits = nn.Linear(width, 4 * self.num_bins)
        actor = {k: v for k, v in weights.items()
                 if k.startswith('actor_') and not k.startswith('actor_aux_head.')}
        expected = self.state_dict()
        if set(actor) != set(expected) or any(actor[k].shape != v.shape for k, v in expected.items()):
            raise ValueError('unsupported/mismatched actor tensors')
        if any(not torch.isfinite(v).all() for v in actor.values()):
            raise ValueError('nonfinite actor weights')
        self.load_state_dict(actor, strict=True)
        norm = checkpoint.get('norm')
        self.register_buffer('mean', None)
        self.register_buffer('var', None)
        if norm is not None:
            mean, var = torch.as_tensor(norm['mean']), torch.as_tensor(norm['var'])
            if (mean.shape != (214,) or var.shape != (214,) or not torch.isfinite(mean).all()
                    or not torch.isfinite(var).all() or (var < 0).any()):
                raise ValueError('invalid normalization')
            self.mean, self.var = mean.clone(), var.clone()
        self.requires_grad_(False)
        self.eval()

    def initial_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.actor_gru.hidden_size, device=device)
                if self.actor_gru is not None else None)

    @torch.inference_mode()
    def act(self, obs, state, episode_start, sample=False):
        if sample:
            raise ValueError('Junhwa tournament adapter is deterministic only')
        if obs.ndim != 2 or obs.shape[1] != 214:
            raise ValueError('expected batched 214D observations')
        if self.mean is not None:
            obs = ((obs.to(self.mean.dtype) - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)
        x = self.actor_body(obs.float())
        if self.actor_gru is not None:
            state = state * (~episode_start.bool()).to(state.dtype).reshape(-1, 1)
            state = self.actor_gru(x, state)
            x = state
        return self.actor_logits(x).reshape(-1, 4, self.num_bins).argmax(-1), state


def load_junhwa_policy(path, device='cuda'):
    # Never deserialize executable custom objects from teammates' checkpoints.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    return JunhwaPolicy(checkpoint).to(device), None
