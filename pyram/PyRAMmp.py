''' PyRAMmp class definition '''

from multiprocessing.pool import Pool
from pyram.PyRAM import PyRAM
from time import sleep


class PyRAMmp():

    '''
    The PyRAMmp class sets up and runs a multiprocessing pool to enable
    parallel PyRAM model runs.
    '''

    def __init__(self, processes=None, maxtasksperchild=None):

        '''
        Initialise the pool and variable lists.
        processes and maxtasksperchild are passed to the pool.
        '''

        self.pool = Pool(processes=processes, maxtasksperchild=maxtasksperchild)
        self.results = []  # Results from PyRAM.run()
        self._outputs = []  # New outputs from PyRAM.run() for transfer to self.results
        self._waiting = []  # Waiting runs
        self._num_waiting = 0  # Number of waiting runs
        self._num_active = 0  # Number of active runs
        self._sleep_time = 1e-3
        self._first = True

    def submit_runs(self, runs):

        '''
        Submit new runs to the pool as resources become available
        runs is a list of PyRAM input tuples (args, kwargs)
        '''

        # Add to waiting list
        for run in runs:
            self._waiting.append(run)
        self._num_waiting = len(self._waiting)

        # Check how many active runs have finished
        num_free = 0
        for run in self._outputs:
            self.results.append(run)
            self._outputs.remove(run)
            self._num_active -= 1
            num_free += 1

        # Start new runs if processes are free
        if self._first:
            num_start = self.pool._processes
        else:
            num_start = min(num_free, self._num_waiting)
        for _ in range(num_start):
            run = self._waiting.pop(0)
            self._add_run(run)
            self._num_active += 1

        if self._first:
            self._first = False
            self._wait()

    def _wait(self):

        '''
        Wait for all submitted runs to complete.
        '''

        while self._num_active > 0:
            self.submit_runs([])
            sleep(self._sleep_time)

    def close(self):

        '''
        Close the pool and wait for all processes to finish.
        '''

        self.pool.close()
        self.pool.join()

    def _add_run(self, run):

        '''
        Add a new PyRAM run.
        '''

        args, kwargs = run[0], run[1]

        freq = args['freq']
        zs = args['zs']
        zr = args['zr']
        z_ss = args['z_ss']
        rp_ss = args['rp_ss']
        cw = args['cw']
        z_sb = args['z_sb']
        rp_sb = args['rp_sb']
        cb = args['cb']
        rhob = args['rhob']
        attn = args['attn']
        rbzb = args['rbzb']

        pyram = PyRAM(freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob,
                      attn, rbzb, **kwargs)

        self.pool.apply_async(pyram.run, callback=self._get_output)

    def _get_output(self, output):

        '''
        Get a PyRAM output.
        '''

        self._outputs.append(output)

    def __del__(self):

        self.close()
