'''
TestPyRAMmp unit test class.
Uses configuration file TestPyRAMmp_Config.xml.
Computational range and depth steps and number of repetitions are configurable.
Number of PyRAM runs = 5 * number of repetitions.
Tests should always pass but speedup will depend upon computing environment.
'''

import unittest
import numpy
from pyram.PyRAMmp import PyRAMmp
from pyram.PyRAM import PyRAM
import xml.etree.ElementTree as et
from time import time


class TestPyRAMmp(unittest.TestCase):

    '''
    Test PyRAMmp using the test case supplied with RAM.
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
                nrep = int(child.text)

        self.test_freq = 50

        self.pyram_args = dict(freq=self.test_freq,
                               zs=50,
                               zr=50,
                               z_ss=numpy.array([0, 100, 400]),
                               rp_ss=numpy.array([0, 25000]),
                               cw=numpy.array([[1480, 1530],
                                               [1520, 1530],
                                               [1530, 1530]]),
                               z_sb=numpy.array([0]),
                               rp_sb=numpy.array([0]),
                               cb=numpy.array([[1700]]),
                               rhob=numpy.array([[1.5]]),
                               attn=numpy.array([[0.5]]),
                               rbzb=numpy.array([[0, 200],
                                                 [40000, 400]]))

        self.pyram_kwargs = dict(rmax=50000,
                                 dr=dr,
                                 dz=dz,
                                 zmplt=500,
                                 c0=1600)

        pyram = PyRAM(self.pyram_args['freq'], self.pyram_args['zs'],
                      self.pyram_args['zr'], self.pyram_args['z_ss'],
                      self.pyram_args['rp_ss'], self.pyram_args['cw'],
                      self.pyram_args['z_sb'], self.pyram_args['rp_sb'],
                      self.pyram_args['cb'], self.pyram_args['rhob'],
                      self.pyram_args['attn'], self.pyram_args['rbzb'],
                      **self.pyram_kwargs)

        results = pyram.run()

        self.ref_r = results['Ranges']
        self.ref_tl = results['TL Line']

        self.freqs = numpy.tile([self.test_freq-20,
                                 self.test_freq-10,
                                 self.test_freq,
                                 self.test_freq+10,
                                 self.test_freq+20], nrep)  # nrep*5 runs

        self.tl_tol = 1e-2  # Tolerable mean difference in TL (dB)

    def tearDown(self):
        pass

    def test_PyRAM(self):

        num_runs = len(self.freqs)

        print(num_runs, 'PyRAM runs set up, running...', )

        runs = []
        for n in range(num_runs):
            pyram_args = self.pyram_args.copy()
            pyram_args['freq'] = self.freqs[n]
            pyram_kwargs = self.pyram_kwargs.copy()
            pyram_kwargs['id'] = n
            runs.append((pyram_args, pyram_kwargs))

        pyram_mp = PyRAMmp()
        nproc = pyram_mp.pool._processes
        t0 = time()
        pyram_mp.submit_runs(runs[:int(num_runs/2)])  # Submit in 2 batches
        pyram_mp.submit_runs(runs[int(num_runs/2):])
        self.elap_time = time() - t0  # Approximate value as process_time can't be used

        results = [None]*num_runs
        self.proc_time = 0
        for result in pyram_mp.results:
            rid = result['ID']
            results[rid] = result
            self.proc_time += result['Proc Time']

        pyram_mp.close()

        for n in range(num_runs):

            self.assertTrue(numpy.array_equal(self.ref_r, results[n]['Ranges']),
                            'Ranges are not equal')

            mean_diff = numpy.mean(numpy.abs(results[n]['TL Line'] - self.ref_tl))
            if runs[n][0]['freq'] == self.test_freq:
                self.assertTrue(mean_diff <= self.tl_tol,
                                'Mean TL difference not within tolerance for test frequency')
            else:
                self.assertTrue(mean_diff > self.tl_tol,
                                'Mean TL difference within tolerance for non-test frequency')

        print('Finished.\n')
        speed_fact = 100*(self.proc_time/nproc)/self.elap_time
        print('{0:.1f} % of expected speed up achieved'.format(speed_fact))

if __name__ == "__main__":
    unittest.main()
