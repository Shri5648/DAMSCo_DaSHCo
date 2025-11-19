import torch
import torch.nn.functional as F
from torch import Tensor
from src.Compressor import NoneCompressor
from src.Optimizer import Optimizer

@torch.compile
def SVD_exact(G: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Compute full SVD of the gradient tensor
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    #print(f'Singular Value Matrix={S}')
    return U, S, Vh

class MuonexactSVD(Optimizer):
    def __init__(self,params,lr=0.05, weight_decay=0.01, momentum=0.95, compressor=NoneCompressor(),device="cpu",devices=[],comm_set=['x'],lr_decay="none",nvlink=False):
        print('lr=,weight decay=,momentum=',lr,weight_decay,momentum)
        # Ensure params is a list
        if not isinstance(params, list):
            params = list(params)

        super().__init__(params,compressor=compressor,optim_name="ESMuon",comm_set=comm_set,device=device,topology="ring",devices=devices,
                         nvlink=nvlink,lr_decay=lr_decay,lr=lr)

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.epoch = 0
                
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
                
                # Compute SVD of gradient
                U, S, Vh = SVD_exact(grad)

                # Compute modified singular values: S^{1/p - 1}
                #exp = 1.0 /(pval - 1.0)
                #S_mod = torch.pow(S, exp)

                # Reconstruct search direction d = -U · diag(S_mod) · Vh
                d = -U.matmul(Vh)

                # Update parameters along custom SVD direction
                p.add_(d, alpha=group['lr'])


    def state_dict(self):
        state = {
            'optim_name': self.optim_name,
            'steps': self.steps,
            'epoch': self.epoch,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'lr': self.lr
        }
        return state
