'''solve function definition'''

from numba import jit, float64, int64, complex128


@jit((float64, float64, int64, int64, int64, int64, float64[:], float64[:],
     float64[:], complex128[:], float64[:], float64[:], float64[:],
     complex128[:], float64[:], complex128[:, :], complex128[:, :],
     complex128[:, :], complex128[:, :], complex128[:, :], complex128[:, :],
     complex128[:], complex128[:]), nopython=True)
def matrc(k0, dz, iz, jz, nz, np, f1, f2, f3, ksq, alpw, alpb, ksqw, ksqb,
          rhob, r1, r2, r3, s1, s2, s3, pd1, pd2):

    '''The tridiagonal matrices'''

    a1 = k0**2/6
    a2 = 2*k0**2/3
    a3 = a1
    cfact = 0.5/dz**2
    dfact = 1/12

    # New matrices when iz == jz
    if iz == jz:
        i1 = 1
        i2 = nz
        for i in range(iz+1):
            f1[i] = 1/alpw[i]
            f2[i] = 1
            f3[i] = alpw[i]
            ksq[i] = ksqw[i]
        for i in range(iz+1, nz+2):
            f1[i] = rhob[i]/alpb[i]
            f2[i] = 1/rhob[i]
            f3[i] = alpb[i]
            ksq[i] = ksqb[i]
    # Updated matrices when iz != jz
    elif iz > jz:
        i1 = jz
        i2 = iz+1
        for i in range(jz+1, iz+1):
            f1[i] = 1/alpw[i]
            f2[i] = 1
            f3[i] = alpw[i]
            ksq[i] = ksqw[i]
    elif iz < jz:
        i1 = iz
        i2 = jz+1
        for i in range(iz+1, jz+1):
            f1[i] = rhob[i]/alpb[i]
            f2[i] = 1/rhob[i]
            f3[i] = alpb[i]
            ksq[i] = ksqb[i]

    # Discretization by Galerkin's method

    for i in range(i1, i2+1):

        c1 = cfact*f1[i]*(f2[i-1] + f2[i])*f3[i-1]
        c2 = -cfact*f1[i]*(f2[i-1] + 2*f2[i] + f2[i+1])*f3[i]
        c3 = cfact*f1[i]*(f2[i] + f2[i+1])*f3[i+1]
        d1 = c1 + dfact*(ksq[i-1] + ksq[i])
        d2 = c2 + dfact*(ksq[i-1] + 6*ksq[i] + ksq[i+1])
        d3 = c3 + dfact*(ksq[i] + ksq[i+1])

        for j in range(np):
            r1[i, j] = a1 + pd2[j]*d1
            r2[i, j] = a2 + pd2[j]*d2
            r3[i, j] = a3 + pd2[j]*d3
            s1[i, j] = a1 + pd1[j]*d1
            s2[i, j] = a2 + pd1[j]*d2
            s3[i, j] = a3 + pd1[j]*d3

    # The matrix decomposition
    for j in range(np):

        for i in range(i1, iz+1):
            rfact = 1/(r2[i, j] - r1[i, j]*r3[i-1, j])
            r1[i, j] *= rfact
            r3[i, j] *= rfact
            s1[i, j] *= rfact
            s2[i, j] *= rfact
            s3[i, j] *= rfact

        for i in range(i2, iz+1, -1):
            rfact = 1/(r2[i, j] - r3[i, j]*r1[i+1, j])
            r1[i, j] *= rfact
            r3[i, j] *= rfact
            s1[i, j] *= rfact
            s2[i, j] *= rfact
            s3[i, j] *= rfact

        r2[iz+1, j] -= r1[iz+1, j]*r3[iz, j]
        r2[iz+1, j] -= r3[iz+1, j]*r1[iz+2, j]
        r2[iz+1, j] = 1/r2[iz+1, j]
