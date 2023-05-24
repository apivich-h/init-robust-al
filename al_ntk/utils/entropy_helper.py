import jax.numpy as jnp

from al_ntk.utils.linalg_jax import symmetric_block_inverse
from al_ntk.utils.kernels_helper import compute_kernel_in_batches, compute_diag_kernel_in_batches


def max_entropy_selector(cov: jnp.ndarray, m: int, entropy_lb: float = 0.):
    idxs = []
    curr_best = -jnp.inf
    v_size = jnp.diag(cov)
    for i in range(m):
        if i == 0:
            val = v_size
            best = jnp.argmax(val)
            A_inv = jnp.linalg.inv(cov[[best], :][:, [best]])
        else:
            b = cov[idxs, :]
            val = v_size - jnp.sum(b * (A_inv @ b), axis=0)
            best = jnp.argmax(val)
            A_inv = symmetric_block_inverse(A_inv=A_inv, B=cov[idxs, :][:, [best]],
                                            D=cov[[best], :][:, [best]])
        idxs.append(int(best))
        h = jnp.linalg.slogdet(cov[idxs, :][:, idxs])[1]
        if h < entropy_lb <= 0.:
            idxs = idxs[:-1]
            break
        else:
            curr_best = h
    return sorted(idxs)


def max_entropy_selector_from_fn(kernel_fn, xs, m: int, entropy_lb: float = 0.):
    idxs = []
    curr_best = -jnp.inf
    v_size = kernel_fn(x1=xs, x2=None, get_diagonal_only=True)
    for i in range(m):
        if i == 0:
            val = v_size
            best = jnp.argmax(val)
            A_inv = jnp.linalg.inv(val[best].reshape(1, 1))
        else:
            b = kernel_fn(x1=xs[idxs, :], x2=xs)
            val = v_size - jnp.sum(b * (A_inv @ b), axis=0)
            val = val.at[jnp.array(idxs)].set(-jnp.inf)
            best = jnp.argmax(val)
            B = b[:, best:best+1]
            D = v_size[best].reshape(1, 1)
            A_inv = symmetric_block_inverse(A_inv=A_inv, B=B, D=D)
        idxs.append(int(best))
        h = -jnp.linalg.slogdet(A_inv)[1]
        if h < entropy_lb <= 0.:
            idxs = idxs[:-1]
            break
        else:
            curr_best = h
    return sorted(idxs)
