from functools import partial
from typing import Callable

import jax
from jax import numpy as jnp
from jax.scipy.linalg import block_diag
import neural_tangents as nt


def generate_batched_kernel(kernel_fn, returns_diagonal: bool = False, 
                            batch_sz: int = 128, device_count: int = -1, store_on_device: bool = False):
    
    _batched_kernel = nt.batch(kernel_fn=kernel_fn,
                               batch_size=batch_sz,
                               device_count=device_count,
                               store_on_device=store_on_device)
    
    device_count = jax.local_device_count()
    divisor_size = device_count * batch_sz
    
    def _kernel(x1, x2=None, **kwargs):
        input_shape = x1.shape[1:]
        n1 = x1.shape[0]
        n2 = n1 if (x2 is None) else x2.shape[0]
        
        # if the arrays are all small enough
        if (n1 <= batch_sz) and (n2 <= batch_sz):
            return kernel_fn(x1, x2=x2, **kwargs)
        
        else:
            # extra padding
            n1_extra = divisor_size - (n1 % divisor_size)
            n2_extra = divisor_size - (n2 % divisor_size)
            
            x1_padded = jnp.concatenate((x1, jnp.empty(shape=(n1_extra, *input_shape))), axis=0)
            x2_padded = (None if x2 is None 
                         else jnp.concatenate((x2, jnp.empty(shape=(n2_extra, *input_shape))), axis=0))
            
            cov = _batched_kernel(x1_padded, x2=x2_padded, **kwargs)
            
            if returns_diagonal:
                assert len(cov.shape) == 1
                return cov[:n1]
            else:
                assert len(cov.shape) == 2
                return cov[:n1][:, :n2]
        
    return _kernel


def compute_kernel_in_batches(kernel_fn, batch_sz=256):
    
    def _kernel(x1, x2=None, **kwargs):
        
        if ((x1.shape[0] == 0) or ((x2 is not None) and (x2.shape[0] == 0)) or 
            (x1.shape[0] < batch_sz) and ((x2 is None) or (x2.shape[0] < batch_sz))):
            return kernel_fn(x1, x2, **kwargs)
        
        if x2 is None:
            symmetric_trick = True
            x2 = x1
        else:
            symmetric_trick = False
                
        idxs = list(range(0, x1.shape[0] + batch_sz, batch_sz))
        idxs[-1] = x1.shape[0]
        jdxs = list(range(0, x2.shape[0] + batch_sz, batch_sz))
        jdxs[-1] = x2.shape[0]
        blocks = []
        for i in range(len(idxs)-1):
            row_block = []
            for j in range(len(jdxs)-1):
                if (not symmetric_trick) or (i <= j):
                    il = idxs[i]
                    iu = idxs[i+1]
                    jl = jdxs[j]
                    ju = jdxs[j+1]
                    row_block.append(kernel_fn(x1[il:iu, :], x2[jl:ju, :]), **kwargs)
                else:
                    # in the case of symmetric matrix, can just reuse values
                    row_block.append(blocks[j][i].T)
            blocks.append(row_block)
        
        mat = jnp.block(blocks)
        return mat
    
    return _kernel


def compute_diag_kernel_in_batches(kernel_fn, batch_sz=256):
    
    def _kernel(x1, x2=None, **kwargs):
        if x1.shape[0] < batch_sz:
            return jnp.diag(kernel_fn(x1, x2=None, **kwargs))
        idxs = list(range(0, x1.shape[0] + batch_sz, batch_sz))
        idxs[-1] = x1.shape[0]
        row_block = []
        for i in range(len(idxs)-1):
            il = idxs[i]
            iu = idxs[i+1]
            ki = jnp.diag(kernel_fn(x1[il:iu, :], x2=None, **kwargs))
            if len(ki.shape) == 1:
                row_block.append(ki)
            else:
                assert ki.shape[0] == ki.shape.shape[1]
                row_block.append(jnp.diag(ki))
        return jnp.concatenate(row_block)
    
    return _kernel


def approximate_full_kernel_as_block_diag(kernel_fn, diag_batch_sz=1024):
    
    def _kernel(x1, x2=None, get_diagonal_only=False):
        
        if (x2 is not None) or (x1.shape[0] < diag_batch_sz) or get_diagonal_only:
            return kernel_fn(x1=x1, x2=x2, get_diagonal_only=get_diagonal_only)
        
        else:
            idxs = list(range(0, x1.shape[0] + diag_batch_sz, diag_batch_sz))
            idxs[-1] = x1.shape[0]
            blocks = []
            for i in range(len(idxs)-1):
                il = idxs[i]
                iu = idxs[i+1]
                blocks.append(kernel_fn(x1=x1[il:iu, :], x2=None, get_diagonal_only=False))
            return block_diag(*blocks)
    
    return _kernel
