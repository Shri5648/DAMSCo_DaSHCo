import torch
import torch.nn.functional as F
from torch import Tensor
from src.Compressor import NoneCompressor

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

class Muon(torch.optim.Optimizer):
    def __init__(self,model,lr=0.001, weight_decay=0.01, momentum=0.95, compressor=NoneCompressor(),device="cpu",devices=[],comm_set=['x'],lr_decay="none",nvlink=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, compressor=compressor, device=device, devices=devices,comm_set=comm_set,lr_decay=lr_decay,nvlink=nvlink)
        super().__init__(model, defaults)
    
    @torch.no_grad()
    def step(self):
        hidden_matrix_params = [p for p in self.model.parameters() if p.ndim >= 2]
        #for group in self.model.param_groups:
        for p in hidden_matrix_params:
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
