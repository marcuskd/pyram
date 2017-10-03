'''solve function definition'''

from numba import jit, int64, complex128


@jit((complex128[:], complex128[:], complex128[:, :], complex128[:, :],
      complex128[:, :], complex128[:, :], complex128[:, :], complex128[:, :],
      int64, int64, int64), nopython=True)
def solve(u, v, s1, s2, s3, r1, r2, r3, iz, nz, np):

    '''The tridiagonal solver'''

    eps = 1e-30

    for j in range(np):
        # The right side
        for i in range(1, nz+1):
            v[i] = s1[i, j]*u[i-1] + s2[i, j]*u[i] + s3[i, j]*u[i+1] + eps

        # The elimination steps
        for i in range(2, iz+1):
            v[i] -= r1[i, j]*v[i-1] + eps
        for i in range(nz-1, iz+1, -1):
            v[i] -= r3[i, j]*v[i+1] + eps

        u[iz+1] = (v[iz+1] - r1[iz+1, j]*v[iz] - r3[iz+1, j]*v[iz+2]) * \
            r2[iz+1, j] + eps

        # The back substitution steps
        for i in range(iz, -1, -1):
            u[i] = v[i] - r3[i, j]*u[i+1] + eps
        for i in range(iz+2, nz+1):
            u[i] = v[i] - r1[i, j]*u[i-1] + eps
