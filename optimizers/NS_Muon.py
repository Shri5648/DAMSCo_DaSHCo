import torch
import torch.nn.functional as F
from torch import Tensor
from src.Compressor import NoneCompressor
from src.Optimizer import Optimizer

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration for matrix orthogonalization"""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.type_as(G)

class Muon(Optimizer):
    def __init__(self,params,lr=0.05, weight_decay=0.01, momentum=0.95, compressor=NoneCompressor(),device="cpu",devices=[],comm_set=['x'],lr_decay="none",nvlink=False):
        print('lr=,weight decay=,momentum=',lr,weight_decay,momentum)
        # Ensure params is a list
        if not isinstance(params, list):
            params = list(params)
        
        
        super().__init__(params,compressor=compressor,optim_name="NSMuon",comm_set=comm_set,device=device,topology="ring",devices=devices,
                         nvlink=nvlink,lr_decay=lr_decay,lr=lr)

        self.weight_decay = weight_decay
        self.momentum = momentum
        if not hasattr(self, 'param_groups') or len(self.param_groups) == 0:
            self.param_groups = [{
                'params': params,
                'lr': lr,
                'weight_decay': weight_decay,
                'momentum': momentum
            }]
        else:
            # If parent created param_groups, ensure hyperparams exist
            for group in self.param_groups:
                group.setdefault('lr', lr)
                group.setdefault('weight_decay', weight_decay)
                group.setdefault('momentum', momentum)

        if not hasattr(self, 'state'):
            self.state = {}

        # Pre-populate state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {}
                
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                momentum_buffer = state['momentum_buffer']
                p.mul_(1 - group['lr'] * group['weight_decay'])
                momentum_buffer.lerp_(grad, 1 - group['momentum'])
                grad = grad.lerp_(momentum_buffer, group['momentum'])
                
                # Apply Newton-Schulz orthogonalization
                v = zeropower_via_newtonschulz5(grad, 5)
                p.add_(v, alpha=-group['lr'])


    def state_dict(self):
        state.update({
            'optim_name': self.optim_name,
            'steps': self.steps,
            'epoch': self.epoch,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'lr': self.lr
        })
        return state
