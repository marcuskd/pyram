'''outpt function definition'''

import numpy
from numba import jit, int64, float64, complex128


@jit(int64[:](float64, int64, int64, int64, int64, float64[:], complex128[:],
              float64, int64, float64[:], float64[:, :]), nopython=True)
def outpt(r, mdr, ndr, ndz, tlc, f3, u, _dir, ir, tll, tlg):

    '''Output transmission loss'''

    eps = 1e-20

    mdr += 1
    if mdr == ndr:
        mdr = 0
        tlc += 1
        ur = (1 - _dir)*f3[ir]*u[ir] + \
            _dir*f3[ir+1]*u[ir+1]
        temp = 10*numpy.log10(r + eps)
        tll[tlc] = -20*numpy.log10(numpy.abs(ur) + eps) + temp

        for i in range(tlg.shape[0]):
            j = (i+1)*ndz - 1
            ur = u[j]*f3[j]
            tlg[i, tlc] = \
                -20*numpy.log10(numpy.abs(ur) + eps) + temp

    return numpy.array([mdr, tlc], dtype=numpy.int64)
