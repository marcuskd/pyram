'''outpt function definition'''

import numpy
from numba import jit, int64, float64, complex128


@jit(int64[:](float64, int64, int64, int64, int64, float64[:],
              complex128[:], float64, int64, float64[:], float64[:, :],
              complex128[:], complex128[:, :]), nopython=True)
def outpt(r, mdr, ndr, ndz, tlc, f3, u, _dir, ir, tll, tlg, cpl, cpg):

    '''
    Output transmission loss and complex pressure.
    Complex pressure does not include cylindrical spreading term 1/sqrt(r)
    or phase term exp(-j*k0*r).
    '''

    eps = 1e-20

    mdr += 1
    if mdr == ndr:
        mdr = 0
        tlc += 1
        cpl[tlc] = (1 - _dir) * f3[ir] * u[ir] + \
            _dir * f3[ir + 1] * u[ir + 1]
        temp = 10 * numpy.log10(r + eps)
        tll[tlc] = -20 * numpy.log10(numpy.abs(cpl[tlc]) + eps) + temp

        for i in range(tlg.shape[0]):
            j = (i + 1) * ndz
            cpg[i, tlc] = u[j] * f3[j]
            tlg[i, tlc] = \
                -20 * numpy.log10(numpy.abs(cpg[i, tlc]) + eps) + temp

    return numpy.array([mdr, tlc], dtype=numpy.int64)
