import jax
import jax.numpy as jnp


@jax.jit
def symmetric_block_inverse_as_blocks(A_inv, B, D):
    """
    Compute the inverse of [[A, B], [B.T, D]] given inverse of A, but return as block
    """
    diag_block_inv = jnp.linalg.inv(D - B.T @ A_inv @ B)
    A_inv_B = A_inv @ B
    return (A_inv + A_inv_B @ diag_block_inv @ A_inv_B.T,
            -A_inv_B @ diag_block_inv,
            diag_block_inv)


@jax.jit
def symmetric_block_inverse_as_blocks_difference(A_inv, B, D):
    """
    Compute the inverse of [[A, B], [B.T, D]] given inverse of A, but return as block
    where A block does not have the inverse term
    """
    diag_block_inv = jnp.linalg.inv(D - B.T @ A_inv @ B)
    A_inv_B = A_inv @ B
    return (A_inv_B @ diag_block_inv @ A_inv_B.T,  # remove the A_inv for simplicity in some cases
            -A_inv_B @ diag_block_inv,
            diag_block_inv)


@jax.jit
def symmetric_block_inverse(A_inv, B, D):
    """
    Compute the inverse of [[A, B], [B.T, D]] given inverse of A (A_inv)
    """
    A_block, B_block, D_block = symmetric_block_inverse_as_blocks(A_inv, B, D)
    return jnp.block([
        [A_block, B_block],
        [B_block.T, D_block]
    ])


@jax.jit
def symmetric_matrix_sum_inverse(A_inv, B, D_inv):
    """
    Compute the inverse of (A + B @ D @ B.T) given inverse of A and inverse of D
    """
    A_inv_B = A_inv @ B
    return A_inv - A_inv_B @ jnp.linalg.inv(D_inv + B.T @ A_inv_B) @ A_inv_B.T


@jax.jit
def diag_matmul(A, B):
    return jnp.einsum('ij,ji->i', A, B)
