'''
PyRAM: Python adaptation of the Range-dependent Acoustic Model (RAM).
RAM was created by Michael D Collins at the US Naval Research Laboratory.
This adaptation is of RAM v1.5, available from the Ocean Acoustics Library at
http://oalib.hlsresearch.com/PE/RAM/

The purpose of PyRAM is to provide a version of RAM which can be used within a
Python interpreter environment (e.g. Spyder or the Jupyter notebook) and is
easier to understand, extend and integrate into other applications than the
Fortran version. It is written in pure Python and achieves speeds comparable to
native code by using the Numba library for JIT compilation.

The PyRAM class contains methods which largely correspond to the original
Fortran subroutines and functions (including retaining the same names). The
variable names are also mostly the same. However some of the original code
(e.g. subroutine zread) is unnecessary when the same purpose can be achieved
using available Python library functions (e.g. from NumPy or SciPy) and has
therefore been replaced.

A difference in functionality is that sound speed profile updates with range
are decoupled from seabed parameter updates, which provides more flexibility
in specifying the environment (e.g. if the data comes from different sources).

PyRAM also provides various conveniences, e.g. automatic calculation of range
and depth steps (though these can be overridden using keyword arguments).
'''

import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from pyram.matrc import matrc
from pyram.solve import solve
from pyram.outpt import outpt


class PyRAM:

    _np_default = 8
    _dzf = 0.1
    _ndr_default = 1
    _ndz_default = 1
    _ns_default = 1
    _lyrw_default = 20
    _id_default = 0

    def __init__(self, freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob,
                 attn, rbzb, **kwargs):

        '''
        -------
        args...
        -------
        freq: Frequency (Hz).
        zs: Source depth (m).
        zr: Receiver depth (m).
        z_ss: Water sound speed profile depths (m), NumPy 1D array.
        rp_ss: Water sound speed profile update ranges (m), NumPy 1D array.
        cw: Water sound speed values (m/s),
            Numpy 2D array, dimensions z_ss.size by rp_ss.size.
        z_sb: Seabed parameter profile depths (m), NumPy 1D array.
        rp_sb: Seabed parameter update ranges (m), NumPy 1D array.
        cb: Seabed sound speed values (m/s),
            NumPy 2D array, dimensions z_sb.size by rp_sb.size.
        rhob: Seabed density values (g/cm3), same dimensions as cb
        attn: Seabed attenuation values (dB/wavelength), same dimensions as cb
        rbzb: Bathymetry (m), Numpy 2D array with columns of ranges and depths
        ---------
        kwargs...
        ---------
        np: Number of Pade terms. Defaults to _np_default.
        c0: Reference sound speed (m/s). Defaults to mean of 1st profile.
        dr: Calculation range step (m). Defaults to np times the wavelength.
        dz: Calculation depth step (m). Defaults to _dzf*wavelength.
        ndr: Number of range steps between outputs. Defaults to _ndr_default.
        ndz: Number of depth steps between outputs. Defaults to _ndz_default.
        zmplt: Maximum output depth (m). Defaults to maximum depth in rbzb.
        rmax: Maximum calculation range (m). Defaults to max in rp_ss or rp_sb.
        ns: Number of stability constraints. Defaults to _ns_default.
        rs: Maximum range of the stability constraints (m). Defaults to rmax.
        lyrw: Absorbing layer width (wavelengths). Defaults to _lyrw_default.
        NB: original zmax input not needed due to lyrw.
        id: Integer identifier for this instance.
        '''

        self._freq, self._zs, self._zr = freq, zs, zr
        self.check_inputs(z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb)
        self.get_params(**kwargs)

    def run(self):

        '''
        Run the model. Sets the following instance variables:
        vr: Calculation ranges (m), NumPy 1D array.
        vz: Calculation depths (m), NumPy 1D array.
        tll: Transmission loss (dB) at receiver depth (zr),
             NumPy 1D array, length vr.size.
        tlg: Transmission loss (dB) grid,
             NumPy 2D array, dimensions vz.size by vr.size.
        proc_time: Processing time (s).
        '''

        t0 = process_time()

        self.setup()

        nr = int(np.round(self._rmax / self._dr)) - 1

        for rn in range(nr):

            self.updat()

            solve(self.u, self.v, self.s1, self.s2, self.s3,
                  self.r1, self.r2, self.r3, self.iz, self.nz, self._np)

            self.r = (rn + 2) * self._dr

            self.mdr, self.tlc = \
                (outpt(self.r, self.mdr, self._ndr, self._ndz, self.tlc, self.f3,
                       self.u, self.dir, self.ir, self.tll, self.tlg, self.cpl, self.cpg)[:])

        self.proc_time = process_time() - t0

        results = {'ID': self._id,
                   'Proc Time': self.proc_time,
                   'Ranges': self.vr,
                   'Depths': self.vz,
                   'TL Grid': self.tlg,
                   'TL Line': self.tll,
                   'CP Grid': self.cpg,
                   'CP Line': self.cpl,
                   'c0': self._c0}

        return results

    def check_inputs(self, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb):

        '''Basic checks on dimensions of inputs'''

        self._status_ok = True

        # Source and receiver depths
        if not z_ss[0] <= self._zs <= z_ss[-1]:
            self._status_ok = False
            raise ValueError('Source depth outside sound speed depths')
        if not z_ss[0] <= self._zr <= z_ss[-1]:
            self._status_ok = False
            raise ValueError('Receiver depth outside sound speed depths')
        if self._status_ok:
            self._z_ss = z_ss

        # Water sound speed profiles
        num_depths = self._z_ss.size
        num_ranges = rp_ss.size
        cw_dims = cw.shape
        if (cw_dims[0] == num_depths) and (cw_dims[1] == num_ranges):
            self._rp_ss, self._cw = rp_ss, cw
        else:
            raise ValueError('Dimensions of z_ss, rp_ss and cw must be consistent.')

        # Seabed profiles
        self._z_sb = z_sb
        num_depths = self._z_sb.size
        num_ranges = rp_sb.size
        for prof in [cb, rhob, attn]:
            prof_dims = prof.shape
            if (prof_dims[0] != num_depths) or (prof_dims[1] != num_ranges):
                self._status_ok = False
        if self._status_ok:
            self._rp_sb, self._cb, self._rhob, self._attn = \
                rp_sb, cb, rhob, attn
        else:
            raise ValueError('Dimensions of z_sb, rp_sb, cb, rhob and attn must be consistent.')

        if rbzb[:, 1].max() <= self._z_ss[-1]:
            self._rbzb = rbzb
        else:
            self._status_ok = False
            raise ValueError('Deepest sound speed point must be at or below deepest bathymetry point.')

        # Set flags for range-dependence (water SSP, seabed profile, bathymetry)
        self.rd_ss = True if self._rp_ss.size > 1 else False
        self.rd_sb = True if self._rp_sb.size > 1 else False
        self.rd_bt = True if self._rbzb.shape[0] > 1 else False

    def get_params(self, **kwargs):

        '''Get the parameters from the keyword arguments'''

        self._np = kwargs.get('np', PyRAM._np_default)

        self._c0 = kwargs.get('c0', np.mean(self._cw[:, 0])
                              if len(self._cw.shape) > 1 else
                              np.mean(self._cw))

        self._lambda = self._c0 / self._freq

        # dr and dz are based on 1500m/s to get sensible output steps
        self._dr = kwargs.get('dr', self._np * 1500 / self._freq)
        self._dz = kwargs.get('dz', PyRAM._dzf * 1500 / self._freq)

        self._ndr = kwargs.get('ndr', PyRAM._ndr_default)
        self._ndz = kwargs.get('ndz', PyRAM._ndz_default)

        self._zmplt = kwargs.get('zmplt', self._rbzb[:, 1].max())

        self._rmax = kwargs.get('rmax', np.max([self._rp_ss.max(),
                                                   self._rp_sb.max(),
                                                   self._rbzb[:, 0].max()]))

        self._ns = kwargs.get('ns', PyRAM._ns_default)
        self._rs = kwargs.get('rs', self._rmax + self._dr)

        self._lyrw = kwargs.get('lyrw', PyRAM._lyrw_default)

        self._id = kwargs.get('id', PyRAM._id_default)

        self.proc_time = None

    def setup(self):

        '''Initialise the parameters, acoustic field, and matrices'''

        if self._rbzb[-1, 0] < self._rmax:
            self._rbzb = np.append(self._rbzb,
                                      [[self._rmax, self._rbzb[-1, 1]]],
                                      axis=0)

        self.eta = 1 / (40 * np.pi * np.log10(np.exp(1)))
        self.ib = 0  # Bathymetry pair index
        self.mdr = 0  # Output range counter
        self.r = self._dr
        self.omega = 2 * np.pi * self._freq
        ri = self._zr / self._dz
        self.ir = int(np.floor(ri))  # Receiver depth index
        self.dir = ri - self.ir  # Offset
        self.k0 = self.omega / self._c0
        self._z_sb += self._z_ss[-1]  # Make seabed profiles relative to deepest water profile point
        self._zmax = self._z_sb.max() + self._lyrw * self._lambda
        self.nz = int(np.floor(self._zmax / self._dz)) - 1  # Number of depth grid points - 2
        self.nzplt = int(np.floor(self._zmplt / self._dz))  # Deepest output grid point
        self.iz = int(np.floor(self._rbzb[0, 1] / self._dz))  # First index below seabed
        self.iz = max(1, self.iz)
        self.iz = min(self.nz - 1, self.iz)

        self.u = np.zeros(self.nz + 2, dtype=np.complex128)
        self.v = np.zeros(self.nz + 2, dtype=np.complex128)
        self.ksq = np.zeros(self.nz + 2, dtype=np.complex128)
        self.ksqb = np.zeros(self.nz + 2, dtype=np.complex128)
        self.r1 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.r2 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.r3 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.s1 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.s2 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.s3 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.pd1 = np.zeros(self._np, dtype=np.complex128)
        self.pd2 = np.zeros(self._np, dtype=np.complex128)

        self.alpw = np.zeros(self.nz + 2)
        self.alpb = np.zeros(self.nz + 2)
        self.f1 = np.zeros(self.nz + 2)
        self.f2 = np.zeros(self.nz + 2)
        self.f3 = np.zeros(self.nz + 2)
        self.ksqw = np.zeros(self.nz + 2)
        nvr = int(np.floor(self._rmax / (self._dr * self._ndr)))
        self._rmax = nvr * self._dr * self._ndr
        nvz = int(np.floor(self.nzplt / self._ndz))
        self.vr = np.arange(1, nvr + 1) * self._dr * self._ndr
        self.vz = np.arange(1, nvz + 1) * self._dz * self._ndz
        self.tll = np.zeros(nvr)
        self.tlg = np.zeros([nvz, nvr])
        self.cpl = np.zeros(nvr) * 1j
        self.cpg = np.zeros([nvz, nvr]) * 1j
        self.tlc = -1  # TL output range counter

        self.ss_ind = 0  # Sound speed profile range index
        self.sb_ind = 0  # Seabed parameters range index
        self.bt_ind = 0  # Bathymetry range index

        # The initial profiles and starting field
        self.profl()
        self.selfs()
        self.mdr, self.tlc = \
            (outpt(self.r, self.mdr, self._ndr, self._ndz, self.tlc, self.f3,
                   self.u, self.dir, self.ir, self.tll, self.tlg, self.cpl, self.cpg)[:])

        # The propagation matrices
        self.epade()
        matrc(self.k0, self._dz, self.iz, self.iz, self.nz, self._np,
              self.f1, self.f2, self.f3, self.ksq, self.alpw, self.alpb,
              self.ksqw, self.ksqb, self.rhob, self.r1, self.r2, self.r3,
              self.s1, self.s2, self.s3, self.pd1, self.pd2)

    def profl(self):

        '''Set up the profiles'''

        attnf = 10  # 10dB/wavelength at floor

        z = np.linspace(0, self._zmax, self.nz + 2)
        self.cw = np.interp(z, self._z_ss, self._cw[:, self.ss_ind],
                               left=self._cw[0, self.ss_ind],
                               right=self._cw[-1, self.ss_ind])
        self.cb = np.interp(z, self._z_sb, self._cb[:, self.sb_ind],
                               left=self._cb[0, self.sb_ind],
                               right=self._cb[-1, self.sb_ind])
        self.rhob = np.interp(z, self._z_sb, self._rhob[:, self.sb_ind],
                                 left=self._rhob[0, self.sb_ind],
                                 right=self._rhob[-1, self.sb_ind])
        attnlyr = np.concatenate((self._attn[:, self.sb_ind],
                                     [self._attn[-1, self.sb_ind], attnf]))
        zlyr = np.concatenate((self._z_sb,
                                  [self._z_sb[-1] + 0.75 * self._lyrw * self._lambda,
                                   self._z_sb[-1] + self._lyrw * self._lambda]))
        self.attn = np.interp(z, zlyr, attnlyr,
                                 left=self._attn[0, self.sb_ind],
                                 right=attnf)

        for i in range(self.nz + 2):
            self.ksqw[i] = (self.omega / self.cw[i])**2 - self.k0**2
            self.ksqb[i] = ((self.omega / self.cb[i]) *
                            (1 + 1j * self.eta * self.attn[i]))**2 - self.k0**2
            self.alpw[i] = np.sqrt(self.cw[i] / self._c0)
            self.alpb[i] = np.sqrt(self.rhob[i] * self.cb[i] / self._c0)

    def updat(self):

        '''Matrix updates'''

        # Varying bathymetry
        if self.rd_bt:
            npt = self._rbzb.shape[0]
            while (self.bt_ind < npt - 1) and (self.r >= self._rbzb[self.bt_ind + 1, 0]):
                self.bt_ind += 1
            jz = self.iz
            z = self._rbzb[self.bt_ind, 1] + \
                (self.r + 0.5 * self._dr - self._rbzb[self.bt_ind, 0]) * \
                (self._rbzb[self.bt_ind + 1, 1] - self._rbzb[self.bt_ind, 1]) / \
                (self._rbzb[self.bt_ind + 1, 0] - self._rbzb[self.bt_ind, 0])
            self.iz = int(np.floor(z / self._dz))  # First index below seabed
            self.iz = max(1, self.iz)
            self.iz = min(self.nz - 1, self.iz)
            if (self.iz != jz):
                matrc(self.k0, self._dz, self.iz, jz, self.nz, self._np,
                      self.f1, self.f2, self.f3, self.ksq, self.alpw,
                      self.alpb, self.ksqw, self.ksqb, self.rhob, self.r1,
                      self.r2, self.r3, self.s1, self.s2, self.s3, self.pd1,
                      self.pd2)

        # Varying sound speed profile
        if self.rd_ss:
            npt = self._rp_ss.size
            ss_ind_o = self.ss_ind
            while (self.ss_ind < npt - 1) and (self.r >= self._rp_ss[self.ss_ind + 1]):
                self.ss_ind += 1
            if self.ss_ind != ss_ind_o:
                self.profl()
                matrc(self.k0, self._dz, self.iz, self.iz, self.nz, self._np,
                      self.f1, self.f2, self.f3, self.ksq, self.alpw,
                      self.alpb, self.ksqw, self.ksqb, self.rhob, self.r1,
                      self.r2, self.r3, self.s1, self.s2, self.s3, self.pd1,
                      self.pd2)

        # Varying seabed profile
        if self.rd_sb:
            npt = self._rp_sb.size
            sb_ind_o = self.sb_ind
            while (self.sb_ind < npt - 1) and (self.r >= self._rp_sb[self.sb_ind + 1]):
                self.sb_ind += 1
            if self.sb_ind != sb_ind_o:
                self.profl()
                matrc(self.k0, self._dz, self.iz, self.iz, self.nz, self._np,
                      self.f1, self.f2, self.f3, self.ksq, self.alpw,
                      self.alpb, self.ksqw, self.ksqb, self.rhob, self.r1,
                      self.r2, self.r3, self.s1, self.s2, self.s3, self.pd1,
                      self.pd2)

        # Turn off the stability constraints
        if self.r >= self._rs:
            self._ns = 0
            self._rs = self._rmax + self._dr
            self.epade()
            matrc(self.k0, self._dz, self.iz, self.iz, self.nz, self._np,
                  self.f1, self.f2, self.f3, self.ksq, self.alpw, self.alpb,
                  self.ksqw, self.ksqb, self.rhob, self.r1, self.r2, self.r3,
                  self.s1, self.s2, self.s3, self.pd1, self.pd2)

    def selfs(self):

        '''The self-starter'''

        # Conditions for the delta function

        si = self._zs / self._dz
        _is = int(np.floor(si))  # Source depth index
        dis = si - _is  # Offset

        self.u[_is] = (1 - dis) * np.sqrt(2 * np.pi / self.k0) / \
            (self._dz * self.alpw[_is])
        self.u[_is + 1] = dis * np.sqrt(2 * np.pi / self.k0) / \
            (self._dz * self.alpw[_is])

        # Divide the delta function by (1-X)**2 to get a smooth rhs

        self.pd1[0] = 0
        self.pd2[0] = -1

        matrc(self.k0, self._dz, self.iz, self.iz, self.nz, 1,
              self.f1, self.f2, self.f3, self.ksq, self.alpw, self.alpb,
              self.ksqw, self.ksqb, self.rhob, self.r1, self.r2, self.r3,
              self.s1, self.s2, self.s3, self.pd1, self.pd2)
        for _ in range(2):
            solve(self.u, self.v, self.s1, self.s2, self.s3,
                  self.r1, self.r2, self.r3, self.iz, self.nz, 1)

        # Apply the operator (1-X)**2*(1+X)**(-1/4)*exp(ci*k0*r*sqrt(1+X))

        self.epade(ip=2)
        matrc(self.k0, self._dz, self.iz, self.iz, self.nz, self._np,
              self.f1, self.f2, self.f3, self.ksq, self.alpw, self.alpb,
              self.ksqw, self.ksqb, self.rhob, self.r1, self.r2, self.r3,
              self.s1, self.s2, self.s3, self.pd1, self.pd2)
        solve(self.u, self.v, self.s1, self.s2, self.s3,
              self.r1, self.r2, self.r3, self.iz, self.nz, self._np)

    def epade(self, ip=1):

        '''The coefficients of the rational approximation'''

        n = 2 * self._np
        _bin = np.zeros([n + 1, n + 1])
        a = np.zeros([n + 1, n + 1], dtype=np.complex128)
        b = np.zeros(n, dtype=np.complex128)
        dg = np.zeros(n + 1, dtype=np.complex128)
        dh1 = np.zeros(n, dtype=np.complex128)
        dh2 = np.zeros(n, dtype=np.complex128)
        dh3 = np.zeros(n, dtype=np.complex128)
        fact = np.zeros(n + 1)
        sig = self.k0 * self._dr

        if ip == 1:
            nu, alp = 0, 0
        else:
            nu, alp = 1, -0.25

        # The factorials
        fact[0] = 1
        for i in range(1, n):
            fact[i] = (i + 1) * fact[i - 1]

        # The binomial coefficients
        for i in range(n + 1):
            _bin[i, 0] = 1
            _bin[i, i] = 1
        for i in range(2, n + 1):
            for j in range(1, i):
                _bin[i, j] = _bin[i - 1, j - 1] + _bin[i - 1, j]

        # The accuracy constraints
        dg, dh1, dh2, dh3 = \
            self.deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu)
        for i in range(n):
            b[i] = dg[i + 1]
        for i in range(n):
            if 2 * i <= n - 1:
                a[i, 2 * i] = fact[i]
            for j in range(i + 1):
                if 2 * j + 1 <= n - 1:
                    a[i, 2 * j + 1] = -_bin[i + 1, j + 1] * fact[j] * dg[i - j]

        # The stability constraints

        if self._ns >= 1:
            z1 = -3 + 0j
            b[n - 1] = -1
            for j in range(self._np):
                a[n - 1, 2 * j] = z1**(j + 1)
                a[n - 1, 2 * j + 1] = 0

        if self._ns >= 2:
            z1 = -1.5 + 0j
            b[n - 2] = -1
            for j in range(self._np):
                a[n - 2, 2 * j] = z1**(j + 1)
                a[n - 2, 2 * j + 1] = 0

        a, b = self.gauss(n, a, b, self.pivot)

        dh1[0] = 1
        for j in range(self._np):
            dh1[j + 1] = b[2 * j]
        dh1, dh2 = self.fndrt(dh1, self._np, dh2, self.guerre)
        for j in range(self._np):
            self.pd1[j] = -1 / dh2[j]

        dh1[0] = 1
        for j in range(self._np):
            dh1[j + 1] = b[2 * j + 1]
        dh1, dh2 = self.fndrt(dh1, self._np, dh2, self.guerre)
        for j in range(self._np):
            self.pd2[j] = -1 / dh2[j]

    @staticmethod
    def deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu):

        '''The derivatives of the operator function at x=0'''

        dh1[0] = 0.5 * 1j * sig
        exp1 = -0.5
        dh2[0] = alp
        exp2 = -1
        dh3[0] = -2 * nu
        exp3 = -1
        for i in range(1, n):
            dh1[i] = dh1[i - 1] * exp1
            exp1 -= 1
            dh2[i] = dh2[i - 1] * exp2
            exp2 -= 1
            dh3[i] = -nu * dh3[i - 1] * exp3
            exp3 -= 1

        dg[0] = 1
        dg[1] = dh1[0] + dh2[0] + dh3[0]
        for i in range(1, n):
            dg[i + 1] = dh1[i] + dh2[i] + dh3[i]
            for j in range(i):
                dg[i + 1] += _bin[i, j] * (dh1[j] + dh2[j] + dh3[j]) * dg[i - j]

        return dg, dh1, dh2, dh3

    @staticmethod
    def gauss(n, a, b, pivot):

        '''Gaussian elimination'''

        # Downward elimination
        for i in range(n):
            if i < n - 1:
                a, b = pivot(n, i, a, b)
            a[i, i] = 1 / a[i, i]
            b[i] *= a[i, i]
            if i < n - 1:
                for j in range(i + 1, n + 1):
                    a[i, j] *= a[i, i]
                for k in range(i + 1, n):
                    b[k] -= a[k, i] * b[i]
                    for j in range(i + 1, n):
                        a[k, j] -= a[k, i] * a[i, j]

        # Back substitution
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                b[i] -= a[i, j] * b[j]

        return a, b

    @staticmethod
    def pivot(n, i, a, b):

        '''Rows are interchanged for stability'''

        i0 = i
        amp0 = np.abs(a[i, i])
        for j in range(i + 1, n):
            amp = np.abs(a[j, i])
            if amp > amp0:
                i0 = j
                amp0 = amp

        if i0 != i:
            b[i0], b[i] = b[i], b[i0]
            for j in range(i, n + 1):
                a[i0, j], a[i, j] = a[i, j], a[i0, j]

        return a, b

    @staticmethod
    def fndrt(a, n, z, guerre):

        '''The root finding subroutine'''

        if n == 1:
            z[0] = -a[0] / a[1]
            return a, z

        if n != 2:
            for k in range(n - 1, 1, -1):
                # Obtain an approximate root
                root = 0
                err = 1e-12
                a, root, err = guerre(a, k + 1, root, err, 1000)
                # Refine the root by iterating five more times
                err = 0
                a, root, err = guerre(a, k + 1, root, err, 5)
                z[k] = root
                # Divide out the factor (z-root).
                for i in range(k, -1, -1):
                    a[i] += root * a[i + 1]
                for i in range(k + 1):
                    a[i] = a[i + 1]

        z[1] = 0.5 * (-a[1] + np.sqrt(a[1]**2 - 4 * a[0] * a[2])) / a[2]
        z[0] = 0.5 * (-a[1] - np.sqrt(a[1]**2 - 4 * a[0] * a[2])) / a[2]

        return a, z

    @staticmethod
    def guerre(a, n, z, err, nter):

        '''This subroutine finds a root of a polynomial of degree n > 2 by Laguerre's method'''

        az = np.zeros(n, dtype=np.complex128)
        azz = np.zeros(n - 1, dtype=np.complex128)

        eps = 1e-20
        # The coefficients of p'(z) and p''(z)
        for i in range(n):
            az[i] = (i + 1) * a[i + 1]
        for i in range(n - 1):
            azz[i] = (i + 1) * az[i + 1]

        _iter = 0
        jter = 0  # Missing from original code - assume this is correct
        dz = np.Inf

        while (np.abs(dz) > err) and (_iter < nter - 1):
            p = a[n - 1] + a[n] * z
            for i in range(n - 2, -1, -1):
                p = a[i] + z * p
            if np.abs(p) < eps:
                return a, z, err

            pz = az[n - 2] + az[n - 1] * z
            for i in range(n - 3, -1, -1):
                pz = az[i] + z * pz

            pzz = azz[n - 3] + azz[n - 2] * z
            for i in range(n - 4, -1, -1):
                pzz = azz[i] + z * pzz

            # The Laguerre perturbation
            f = pz / p
            g = f**2 - pzz / p
            h = np.sqrt((n - 1) * (n * g - f**2))
            amp1 = np.abs(f + h)
            amp2 = np.abs(f - h)
            if amp1 > amp2:
                dz = -n / (f + h)
            else:
                dz = -n / (f - h)

            _iter += 1

            # Rotate by 90 degrees to avoid limit cycles

            jter += 1
            if jter == 9:
                jter = 0
                dz *= 1j
            z += dz

            if _iter == 100:
                raise ValueError('Laguerre method not converging. Try a different combination of DR and NP.')

        return a, z, err
    
def plot_ramTL(ram_out, rbzb, zs, rmax, zmplt, freq, Title,  **kwargs):
    
    # Colorbar min and max
    vmin = kwargs.pop('vmin')
    vmax = kwargs.pop('vmax')
    
    # Bathy
    rb = np.array(rbzb[:,0])
    zb = np.array(rbzb[:,1])
            
    # Grid definition
    Zg = np.array(ram_out.tlg)
    Xg = np.array(ram_out.vr)
    Yg = np.array(ram_out.vz)
            
    """ Remove TL in sediment and reduce artifacts """
    for ii,x in enumerate(Xg): # For all map pixels
        for jj,y in enumerate(Yg):
            if y > np.interp(x, rb, zb) or Zg[jj,ii] > vmax or Zg[jj,ii] < vmin or  np.isnan(Zg[jj,ii]):
                Zg[jj,ii] = vmax
    
    # Plot
    Xg, Yg = np.meshgrid(np.array(Xg), Yg)
    fig1, ax1 = plt.subplots()
    im1 = ax1.pcolormesh(Xg/1000,Yg,Zg, cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax)
    ax1.plot(rb/1000, zb, 'k', linewidth=8)
    ax1.scatter(0, zs, label= "Stars", color= "k", s=500, marker="*") 
    ax1.set_xlabel('Range [km]')
    ax1.set_ylabel('Depth [m]')
    ax1.set_title(f"[ RAM-{Title} ] Propagation Loss @ F = {freq} Hz")
    ax1.set_xlim((0, rmax/1000))
    ax1.set_ylim((0, zmplt))
    cbar1 = fig1.colorbar(im1, ax=ax1)
    cbar1.set_label('Loss [dB]', rotation=270, labelpad=15)
    ax1.invert_yaxis()

class ramSolver:
    
    def loadEnv(self, **kwargs):

        self.inputs = dict(freq   = kwargs.pop('freq'),
                           zs     = kwargs.pop('zs'),
                           zr     = kwargs.pop('zr'),
                           z_ss   = kwargs.pop('z_ss'),
                           rp_ss  = kwargs.pop('rp_ss'),
                           cw     = kwargs.pop('cw'),
                           z_sb   = kwargs.pop('z_sb'),
                           rp_sb  = kwargs.pop('rp_sb'),
                           cb     = kwargs.pop('cb'),
                           rhob   = kwargs.pop('rhob'),
                           attn   = kwargs.pop('attn'),
                           rmax   = kwargs.pop('rmax'),
                           dr     = kwargs.pop('dr'),
                           dz     = kwargs.pop('dz'),
                           zmplt  = kwargs.pop('zmplt'),
                           c0     = kwargs.pop('c0'),
                           rbzb   = kwargs.pop('rbzb'))

        self.tl_tol = 1e-2  # Tolerable mean difference in TL (dB) with reference result

    def compute(self, **kwargs):

        self.pyram = PyRAM(self.inputs['freq'], self.inputs['zs'], self.inputs['zr'],
                      self.inputs['z_ss'], self.inputs['rp_ss'], self.inputs['cw'],
                      self.inputs['z_sb'], self.inputs['rp_sb'], self.inputs['cb'],
                      self.inputs['rhob'], self.inputs['attn'], self.inputs['rbzb'],
                      rmax = self.inputs['rmax'], dr=self.inputs['dr'],
                      dz   = self.inputs['dz'], zmplt=self.inputs['zmplt'],
                      c0   = self.inputs['c0'])
        self.pyram.run()
        
        return self.pyram
    

def plot_ramRHOb(rbzb, zs, rmax, zmplt, rp_sb, z_sb, rhob, vmin, vmax, Nxy, Title, **kwargs):
        
    fig2, ax2 = plt.subplots()
            
    Xb = rp_sb
    Yb = z_sb
    Zb = np.array(rhob)
    
    Xg = np.linspace(0, rmax, Nxy)
    Yg = np.linspace(0, zmplt, Nxy)
    Zg = np.zeros([len(Yg), len(Xg)])
    
    # Bathy
    rb = np.array(rbzb[:,0])
    zb = np.array(rbzb[:,1])
    
    # Re-compute map over grid
    for ii,x in enumerate(Xg): # For all map pixels
        for jj,y in enumerate(Yg): 
            
            if y > np.interp(x, rb, zb): # If in sediment (interpolation of bathymetry line between samples)

                # Search for the correct value in sediment profile:
                    
                # Search the last x update point
                x_idx = 0
                while x_idx < len(Xb) and x >= Xb[x_idx]:
                    x_idx += 1
                if x_idx > 0 : 
                    x_idx-=1
                    
                # Search the y nearest value
                idx = 0
                y_idx = 0
                diff = 1e30
                while idx < len(Yb):
                    if np.sqrt((y-Yb[idx])**2) < diff:
                        y_idx = idx
                        diff  = np.sqrt((y-Yb[y_idx])**2)
                    idx += 1
                    
                # Record the value
                Zg[jj,ii] = Zb[y_idx,x_idx]
                
            else: # Else it is in water column
                # Set minimum value
                Zg[jj,ii] = vmin
          
    # Plot
    Xg, Yg = np.meshgrid(Xg/1000, Yg)
    im2 = ax2.pcolormesh(Xg, Yg, Zg, cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax)
    ax2.plot(rb/1000, zb, 'k', linewidth=8)
    ax2.scatter(0, zs, label= "Stars", color= "k", s=500, marker="*") 
    cbar2 = fig2.colorbar(im2, ax=ax2)
    cbar2.set_label('Density [g/cc]', rotation=270, labelpad=15)
    ax2.set_xlabel('Range [km]')
    ax2.set_ylabel('Depth [m]')
    ax2.set_title(f"[ RAM-{Title} ] Density in sediment")
    ax2.invert_yaxis()
    plt.tight_layout()
    return fig2, ax2

def plot_ramSSP(rbzb, zs, rmax, rp_sb, z_sb, rp_ss, z_ss, cw, cb, rhob, Nxy, Title, **pyRAM_settings):
       
    fig2, ax2 = plt.subplots()
            
    X = rp_ss
    Y = z_ss
    Z = np.array(cw)
    
    Xb = rp_sb
    Yb = z_sb
    Zb = np.array(cb)
    
    Xg = np.linspace(0, rmax, Nxy)
    Yg = np.linspace(0, Y[-1], Nxy)
    Zg = np.zeros([len(Yg), len(Xg)])
    
    # Bathy
    rb = np.array(rbzb[:,0])
    zb = np.array(rbzb[:,1])
    
    # Re-compute map over grid
    for ii,x in enumerate(Xg): # For all map pixels
        for jj,y in enumerate(Yg): 
            
            if y > np.interp(x, rb, zb): # If in sediment (interpolation of bathymetry line between samples)

                # Search for the correct value in sediment profile:
                    
                # Search the last x update point
                x_idx = 0
                while x_idx < len(Xb) and x >= Xb[x_idx]:
                    x_idx += 1
                if x_idx > 0 : 
                    x_idx-=1
                    
                # Search the y nearest value
                idx = 0
                y_idx = 0
                diff = 1e30
                while idx < len(Yb):
                    if np.sqrt((y-Yb[idx])**2) < diff:
                        y_idx = idx
                        diff  = np.sqrt((y-Yb[y_idx])**2)
                    idx += 1
                    
                # Record the value
                Zg[jj,ii] = Zb[y_idx,x_idx]
                
            else: # Else it is in water column
                
                # Search for the correct value in water column profile:
                    
                # Search the last x (range) update point
                x_idx = 0
                while x_idx < len(X) and x >= X[x_idx]:
                    x_idx += 1
                if x_idx > 0 : 
                    x_idx-=1
                    
                # Search the y (depth) nearest value
                idx = 0
                y_idx = 0
                diff = 1e30
                while idx < len(Y):
                    if np.sqrt((y-Y[idx])**2) < diff:
                        y_idx = idx
                        diff  = np.sqrt((y-Y[y_idx])**2)
                    idx += 1
                
                # Record the value
                Zg[jj,ii] = Z[y_idx,x_idx]   
          
    # Plot
    Xg, Yg = np.meshgrid(Xg/1000, Yg)
    im2 = ax2.pcolormesh(Xg, Yg, Zg, cmap='jet', shading='gouraud')
    ax2.plot(rb/1000, zb, 'k', linewidth=8)
    ax2.scatter(0, zs, label= "Stars", color= "k", s=500, marker="*") 
    cbar2 = fig2.colorbar(im2, ax=ax2)
    cbar2.set_label('Sound speed [m/s]', rotation=270, labelpad=15)
    ax2.set_xlabel('Range [km]')
    ax2.set_ylabel('Depth[m]')
    ax2.set_title("[RAM: Parabolic Equation] Sound speed profile")
    ax2.invert_yaxis()
    plt.tight_layout()
    
def plot_ramATN(attn, rbzb, zs, rmax, rp_sb, z_sb, zmplt, Nxy, vmin, vmax, Title, **pyRAM_settings):
        
    fig2, ax2 = plt.subplots()
            
    Xb = rp_sb
    Yb = z_sb
    Zb = np.array(attn)
    
    Xg = np.linspace(0, rmax, Nxy)
    Yg = np.linspace(0, zmplt, Nxy)
    Zg = np.zeros([len(Yg), len(Xg)])
    
    # Bathy
    rb = np.array(rbzb[:,0])
    zb = np.array(rbzb[:,1])
    
    # Re-compute map over grid
    for ii,x in enumerate(Xg): # For all map pixels
        for jj,y in enumerate(Yg): 
            
            if y > np.interp(x, rb, zb): # If in sediment (interpolation of bathymetry line between samples)

                # Search for the correct value in sediment profile:
                    
                # Search the last x update point
                x_idx = 0
                while x_idx < len(Xb) and x >= Xb[x_idx]:
                    x_idx += 1
                if x_idx > 0 : 
                    x_idx-=1
                    
                # Search the y nearest value
                idx = 0
                y_idx = 0
                diff = 1e30
                while idx < len(Yb):
                    if np.sqrt((y-Yb[idx])**2) < diff:
                        y_idx = idx
                        diff  = np.sqrt((y-Yb[y_idx])**2)
                    idx += 1
                    
                # Record the value
                Zg[jj,ii] = Zb[y_idx,x_idx]
                
            else: # Else it is in water column
                # Set minimum value
                Zg[jj,ii] = vmin
          
    # Plot
    Xg, Yg = np.meshgrid(Xg/1000, Yg)
    im2 = ax2.pcolormesh(Xg, Yg, Zg, cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax)
    ax2.plot(rb/1000, zb, 'k', linewidth=8)
    ax2.scatter(0, zs, label= "Stars", color= "k", s=500, marker="*") 
    cbar2 = fig2.colorbar(im2, ax=ax2)
    cbar2.set_label('Attenuation [dB/$\lambda$]', rotation=270, labelpad=15)
    ax2.set_xlabel('Range [km]')
    ax2.set_ylabel('Depth [m]')
    ax2.set_title("[ RAM-{Title} ] Attenuation in sediment")
    ax2.invert_yaxis()
    plt.tight_layout()
    
    return fig2, ax2

def plot_ramTline(ram_out, Title, **kwargs):
    
    zr = kwargs.pop('zr')
    freq = kwargs.pop('freq')
    fig0, ax0 = plt.subplots()
    ax0.plot(ram_out.vr/1000, ram_out.tll, 'k')
    ax0.set_xlabel('Range [km]')
    ax0.set_ylabel('Loss [dB re 1m]')
    ax0.set_title(f"[ RAM-{Title} ] Propagation Loss @ z = {zr} m and F = {freq} Hz")
    ax0.grid('all')
    ax0.invert_yaxis()
    
    return fig0, ax0