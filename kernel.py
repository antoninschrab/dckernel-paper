import jax
import jax.numpy as jnp
from jax import vmap, lax


def kernel_matrix(pairwise_matrix, l, kernel, bandwidth):
    """
    Compute kernel matrix for a given kernel and bandwidth. 
    
    Parameters
    ----------
    pairwise_matrix: array_like
        (N, N) matrix of pairwise distances
    l: string
        "l1" or "l2" 
    kernel: str
        "gaussian" or "laplace" or "imq"
    bandwidth: scalar
        positive value for the kernel bandwidth
    
    Returns
    -------
    output: array_like
        (N, N) kernel matrix

    Warning
    -------
    The pair of variables (kernel, l) must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel == "gaussian" and l == "l2":
        return  jnp.exp(-d ** 2)
    elif kernel == "laplace" and l == "l1":
        return  jnp.exp(-d)
    elif kernel == "imq" and l == "l2":
        return (1 + d ** 2) ** (-0.5)
    else:
        raise ValueError(
            'The values of (kernel, l) need to be '
            '("gaussian", "l2") or '
            '("laplace", "l1") or '
            '("imq", "l2").'
        )


def distances(X, Y, l, matrix=False, min_mem=False):
    """
    Compute matrix of pairwise distances. 
    
    Parameters
    ----------
    X: array_like
        The shape of X must be of the form (m, d) where m is the number
        of samples and d is the dimension.
    Y: array_like
        The shape of X must be of the form (n, d) where n is the number
        of samples and d is the dimension.
    l: str
        "l1" or "l2"
    matrix: bool
        If True then output is a (m + n, m + n) matrix.
        If False then output is a ((m + n) * (m + n - 1) / 2, ) vector.
    min_mem: bool
        If True then kernel values are computed sequentially (low memory).
        If False then kernel values are computed together (vectorised, higher memory).
        The speed improvement can vary depending on the use of CPU/GPU.
    
    Returns
    -------
    output: array_like
        if matrix = True output is
            (m + n, m + n) matrix of pairwise distances
        if matrix = False output is
            ((m + n) * (m + n - 1) / 2, ) vector of pairwise distances
            corresponding to the values of the upper triangular matrix
    """
    if min_mem:
        # use scan/map
        if l == "l1":
            dist_vec = lambda y : jnp.sum(jnp.abs(X - y), 1)
        elif l == "l2":
            dist_vec = lambda y : jnp.sqrt(jnp.sum(jnp.square(X - y), 1))
        else:
            raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
        output = lax.map(dist_vec, Y)
    else:
        # use vmap
        if l == "l1":
            dist = lambda x, y : jnp.sum(jnp.abs(x - y))
        elif l == "l2":            
            dist = lambda x, y : jnp.sqrt(jnp.sum(jnp.square(x - y)))
        else:
            raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
        vmapped_dist = vmap(dist, in_axes=(0, None))
        pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
        output = pairwise_dist(X, Y)
    if matrix:
        return output
    else:
        return output[jnp.triu_indices(output.shape[0])]
