import numpy as np


def multidot(X, Y):
    # equivalent to [dot(X[i], Y[i]) for i in range(len(X.shape[0]))]
    return np.sum(X * Y, axis=0)


def inverse_with_check(X):
    try:
        return np.linalg.inv(X)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(X)


def get_basis(K, basis_size):
    # mag = np.diag(K)
    # K = (K.T / mag).T
    basis = np.zeros(shape=K.shape[0], dtype=bool)
    v_size = multidot(K, K)
    best_val = np.inf
    for i in range(basis_size):
        if i == 0:
            val = v_size
        else:
            curr_basis = K[:, basis]
            S = curr_basis @ inverse_with_check(curr_basis.T @ curr_basis) @ curr_basis.T
            val = v_size - multidot(K, S @ K)
        best = np.ma.masked_array(val, mask=basis).argmax()
        if val[best] < best_val:  # check this condition again
            basis[best] = True
            best_val = val[best]
        else:
            break
    return basis
