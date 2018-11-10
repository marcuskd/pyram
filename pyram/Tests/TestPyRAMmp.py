'''
TestPyRAMmp unit test class.
Uses configuration file TestPyRAMmp_Config.xml.
Computational range and depth steps and number of repetitions are configurable.
Number of PyRAM runs = number of frequencies * number of repetitions.
Tests should always pass but speedup will depend upon computing environment.
'''

import unittest
import xml.etree.ElementTree as et
from time import time
from copy import deepcopy
import numpy
from pyram.PyRAMmp import PyRAMmp
from pyram.PyRAM import PyRAM


class TestPyRAMmp(unittest.TestCase):

    '''
    Test PyRAMmp using the test case supplied with RAM and different frequencies.
    '''

    def setUp(self):

        config_file = 'TestPyRAMmp_Config.xml'
        root = et.parse(config_file).getroot()

        for child in root:

            if child.tag == 'RangeStep':
                dr = float(child.text)
            if child.tag == 'DepthStep':
                dz = float(child.text)
            if child.tag == 'NumberOfRepetitions':
                self.nrep = int(child.text)

        self.pyram_args = dict(zs=50.,
                               zr=50.,
                               z_ss=numpy.array([0., 100, 400]),
                               rp_ss=numpy.array([0., 25000]),
                               cw=numpy.array([[1480., 1530],
                                               [1520, 1530],
                                               [1530, 1530]]),
                               z_sb=numpy.array([0.]),
                               rp_sb=numpy.array([0.]),
                               cb=numpy.array([[1700.]]),
                               rhob=numpy.array([[1.5]]),
                               attn=numpy.array([[0.5]]),
                               rbzb=numpy.array([[0., 200],
                                                 [40000, 400]]))

        self.pyram_kwargs = dict(rmax=50000.,
                                 dr=dr,
                                 dz=dz,
                                 zmplt=500.,
                                 c0=1600.)

        self.freqs = [30., 40, 50, 60, 70]

        self.ref_r = []
        self.ref_z = []
        self.ref_tl = []

        for fn in range(len(self.freqs)):

            pyram_args = deepcopy(self.pyram_args)
            pyram_kwargs = deepcopy(self.pyram_kwargs)
            pyram = PyRAM(self.freqs[fn], pyram_args['zs'],
                          pyram_args['zr'], pyram_args['z_ss'],
                          pyram_args['rp_ss'], pyram_args['cw'],
                          pyram_args['z_sb'], pyram_args['rp_sb'],
                          pyram_args['cb'], pyram_args['rhob'],
                          pyram_args['attn'], pyram_args['rbzb'],
                          **pyram_kwargs)

            results = pyram.run()

            self.ref_r.append(results['Ranges'])
            self.ref_z.append(results['Depths'])
            self.ref_tl.append(results['TL Grid'])

    def tearDown(self):
        pass

    def test_PyRAMmp(self):

        '''
        Test that the results from PyRAMmp are the same as from PyRAM. Also measure the speedup.
        '''

        freqs_rep = numpy.tile(self.freqs, self.nrep)
        num_runs = len(freqs_rep)

        print(num_runs, 'PyRAM runs set up, running...', )

        runs = []
        for n in range(num_runs):
            pyram_args = deepcopy(self.pyram_args)
            pyram_args['freq'] = freqs_rep[n]
            pyram_kwargs = deepcopy(self.pyram_kwargs)
            pyram_kwargs['id'] = n
            runs.append((pyram_args, pyram_kwargs))

        pyram_mp = PyRAMmp()
        nproc = pyram_mp.pool._processes
        t0 = time()
        pyram_mp.submit_runs(runs[:int(num_runs / 2)])  # Submit in 2 batches
        pyram_mp.submit_runs(runs[int(num_runs / 2):])
        self.elap_time = time() - t0  # Approximate value as process_time can't be used

        results = [None] * num_runs
        self.proc_time = 0
        for result in pyram_mp.results:
            rid = result['ID']
            results[rid] = result
            self.proc_time += result['Proc Time']

        pyram_mp.close()

        for n in range(num_runs):

            freq = runs[n][0]['freq']
            ind = self.freqs.index(freq)

            self.assertTrue(numpy.array_equal(self.ref_r[ind], results[n]['Ranges']),
                            'Ranges are not equal')
            self.assertTrue(numpy.array_equal(self.ref_z[ind], results[n]['Depths']),
                            'Depths are not equal')
            self.assertTrue(numpy.array_equal(self.ref_tl[ind], results[n]['TL Grid']),
                            'Transmission Loss values are not equal')

        print('Finished.\n')
        speed_fact = 100 * (self.proc_time / nproc) / self.elap_time
        print('{0:.1f} % of expected speed up achieved'.format(speed_fact))

if __name__ == "__main__":
    unittest.main()
