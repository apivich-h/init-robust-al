import random
import torch
from functorch import make_functional, make_functional_with_buffers, vmap, vjp, jvp, jacrev
import jax.numpy as jnp

from al_ntk.utils.kernels_helper import compute_kernel_in_batches, compute_diag_kernel_in_batches


# NTK functions from https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html

def empirical_ntk_jacobian_contraction(func, params, x1, x2=None, only_get_diag=False):
    # Compute J(x1)
    jac1 = vmap(jacrev(func), (None, 0), randomness='same')(params, x1)
    jac1 = [j.flatten(2) for j in jac1]
    
    if only_get_diag or x2 is None:
        jac2 = jac1
    else:
        # Compute J(x2)
        jac2 = vmap(jacrev(func), (None, 0), randomness='same')(params, x2)
        jac2 = [j.flatten(2) for j in jac2]
        
    if only_get_diag:
        einsum_expr = 'Naf,Naf->N'
    else:
        einsum_expr = 'Naf,Maf->NM'
    
    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0) #/ jac1[0].shape[1]  # divide by the output dimension
    return result


def empirical_ntk_ntk_vps(func, params, x1, x2=None, only_get_diag=False):
    
    x2 = x1 if x2 is None else x2
    
    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)

        def func_x2(params):
            return func(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)
        
    # get_ntk(x1, x2) computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to empirical_ntk_ntk_vps are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the vmaps here do.
    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
    
    if only_get_diag:
        einsum_expr = 'NNKK->N'
    else:
        einsum_expr = 'NMKK->NM'
    
    return torch.einsum(einsum_expr, result) #/ result.shape[-1]


def generate_ntk_kernel_torch(model: torch.nn.Module, out_dim: int,
                              method: str = 'ntk_vps', rand_idxs: int = 1):
    
    # remove effects from dropout first
    model.eval()
    fnet, params, buffers = make_functional_with_buffers(model=model, disable_autograd_tracking=False)
        
    if (rand_idxs > out_dim) or (rand_idxs < 1):
        r = out_dim ** -0.5
        fnet_single = lambda params, x: r * fnet(params, buffers, x.unsqueeze(0))
    else:
        idx = torch.tensor(random.sample(range(out_dim), rand_idxs))
        r = rand_idxs ** -0.5
        fnet_single = lambda params, x: r * fnet(params, buffers, x.unsqueeze(0)).squeeze(0)[idx].reshape(rand_idxs)
    
    if method == 'ntk_vps':
        ntk_fn = empirical_ntk_ntk_vps
    elif method == 'jac_con':
        ntk_fn = empirical_ntk_jacobian_contraction
    else:
        raise ValueError
    
    def _kernel(x1: torch.Tensor, x2: torch.Tensor = None, get_diagonal_only: bool = False):
        
        if get_diagonal_only:
            mat = empirical_ntk_jacobian_contraction(
                func=fnet_single,
                params=params,
                x1=x1,
                x2=None,
                only_get_diag=True
            )
            
        else:
            mat = ntk_fn(
                func=fnet_single,
                params=params,
                x1=x1,
                x2=x2,
                only_get_diag=False
            )
            
        return jnp.array(mat.detach().cpu())
            
    return _kernel


def generate_batched_ntk_kernel_torch(model: torch.nn.Module, batch_sz: int = 256, method: str = 'ntk_vps', out_dim: int = None, rand_idxs: int = 1):
    
    _kernel = generate_ntk_kernel_torch(model=model, method=method, out_dim=out_dim, rand_idxs=rand_idxs)
    _kernel_batched_full = compute_kernel_in_batches(kernel_fn=_kernel, batch_sz=batch_sz)
    _kernel_batched_diag = compute_diag_kernel_in_batches(kernel_fn=_kernel, batch_sz=batch_sz)

    
    def _kernel_batched(x1: torch.Tensor, x2: torch.Tensor = None, get_diagonal_only: bool = False):
        
        if x1.shape[0] <= batch_sz and ((x2 is None) or (x2.shape[0] <= batch_sz)):
            return _kernel(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only)
        
        elif get_diagonal_only:
            return _kernel_batched_diag(x1=x1, x2=None)
        
        else:
            return _kernel_batched_full(x1=x1, x2=x2)
        
    return _kernel_batched
