"""Portable frozen inference. Accepts RAW 214D features; normalizes exactly once."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from frozen_actor import JunhwaPolicy


class Policy:
    def __init__(self, path=None, device='cpu'):
        self.device = torch.device(device)
        self.checkpoint = torch.load(path or ROOT/'model.pt', map_location='cpu', weights_only=True)
        self.model = JunhwaPolicy(self.checkpoint).to(self.device).eval()

    def initial_state(self, batch_size=1):
        return self.model.initial_state(batch_size, self.device)

    @torch.inference_mode()
    def act(self, raw_observation, state=None, episode_start=None):
        obs = torch.as_tensor(raw_observation, device=self.device, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2 or obs.shape[1] != 214 or not torch.isfinite(obs).all():
            raise ValueError('Expected finite raw observations of shape (B,214)')
        if state is None:
            state = self.initial_state(len(obs))
        if episode_start is None:
            episode_start = torch.zeros(len(obs), device=self.device, dtype=torch.bool)
        else:
            episode_start = torch.as_tensor(episode_start, device=self.device, dtype=torch.bool).reshape(-1)
        if episode_start.numel() != len(obs):
            raise ValueError('episode_start needs one flag per episode')
        index, state = self.model.act(obs, state, episode_start, sample=False)
        command = torch.linspace(-1.,1.,self.model.num_bins,device=self.device)[index]
        command[:,3] = 0.5*command[:,3]+0.5
        return command.cpu().numpy(), state


class FlightPolicy:
    """Single-aircraft 10 Hz adapter, using the bundled CUDA observation contract."""
    def __init__(self, path=None, device='cpu'):
        from claude_code.observation_contract import CUDAObservationState
        self.policy = Policy(path,device)
        self.observation = CUDAObservationState()
        self.reset()

    def reset(self):
        self.observation.reset()
        self.state = self.policy.initial_state()
        self.first = True

    def command(self, own_state, target_state):
        obs = self.observation.observation(np.asarray(own_state),np.asarray(target_state))
        command,self.state = self.policy.act(obs,self.state,[self.first])
        self.first = False
        self.observation.record_command(command[0])
        return command[0]


if __name__ == '__main__':
    torch.set_num_threads(1)
    p=Policy()
    state=p.initial_state(2)
    for i in range(3):
        commands,state=p.act(np.zeros((2,214),dtype=np.float32),state,[i==0,i==0])
        assert np.isfinite(commands).all()
        assert (np.abs(commands[:,:3])<=1).all() and ((commands[:,3]>=0)&(commands[:,3]<=1)).all()
    print('PASS: model + normalization + recurrent state + command ranges')
