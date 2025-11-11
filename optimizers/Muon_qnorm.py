# For (q<=1)
import torch
import torch.nn.functional as F
from torch import Tensor

@torch.compile
def SVD_exact(G: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Compute full SVD of the gradient tensor
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    #print(f'Singular Value Matrix={S}')
    return U, S, Vh

class Muon_qnorm(torch.optim.Optimizer):
    def __init__(self, params, qval, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,qval=qval)
        print(f'qval={qval}')
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            qval=group['qval']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                    #state['step']=...
                
                momentum_buffer = state['momentum_buffer']
                p.mul_(1 - group['lr'] * group['weight_decay'])
                momentum_buffer.lerp_(grad, 1 - group['momentum'])
                grad = grad.lerp_(momentum_buffer, group['momentum'])

                
                # Compute SVD of gradient
                U, S, Vh = SVD_exact(grad)

                # Compute modified singular values: S^{1/p - 1}
                #exp = 1.0 /(qval - 1.0)
                S_mod = torch.zeros_like(S)
                S_mod[..., 0] = 1.0  # Set first singular value to 1 for all batch elements

                # Reconstruct search direction d = -U · diag(S_mod) · Vh
                d = -U.matmul(torch.diag_embed(S_mod)).matmul(Vh)

                # Update parameters along custom SVD direction
                p.add_(d, alpha=group['lr'])

                 # Increment step counter
                # state['step'] += 1           
               
