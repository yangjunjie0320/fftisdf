import fft.isdf

import os, sys, h5py
import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import lib
from pyscf.lib import logger, current_memory
from pyscf.lib.chkfile import load, dump
from pyscf.lib.logger import process_clock, perf_counter

from pyscf.pbc import tools as pbctools

import fft
from fft.isdf import contract, lstsq
from fft.isdf import get_phase_factor
from fft.isdf import compute_blksize

from pyscf import __config__
from pyscf.pbc.dft.gen_grid import BLKSIZE
CHOLESKY_TOL = getattr(__config__, "fftisdf_cholesky_tol", 1e-20)
CHOLESKY_MAX_SIZE = getattr(__config__, "fftisdf_cholesky_max_size", 20000)
CONTRACT_MAX_SIZE = getattr(__config__, "fftisdf_contract_max_size", 20000)

class WithMPI(fft.isdf.ISDF):
    def __init__(self, cell, kpts, comm=None):
        super().__init__(cell, kpts)
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD

        self._comm = comm
        size = comm.Get_size()
        rank = comm.Get_rank()
        comm.barrier()

        fswap = self._fswap.filename
        self._fswap.close()
        self._fswap = None
        assert not os.path.exists(fswap)
        comm.barrier()

        fswap = comm.bcast(fswap, root=0)
        comm.barrier()

        self._fswap = lib.H5TmpFile(fswap, "w", driver="mpio", comm=comm)
        comm.barrier()

    def _finalize(self):
        log = logger.new_logger(self, self.verbose)
        comm = self._comm
        rank = comm.Get_rank()
        comm.barrier()

        if self._fswap is not None:
            fswap = self._fswap.filename
            self._fswap.close()
            self._fswap = None

        comm.barrier()

        if rank == 0:
            if os.path.exists(fswap):
                os.remove(fswap)

            assert not os.path.exists(fswap)
            log.debug("Successfully removed swap file %s", fswap)

            inpv_kpt = self._inpv_kpt
            coul_kpt = self._coul_kpt
            assert inpv_kpt is not None
            assert coul_kpt is not None

            isdf_to_save = self._isdf_to_save
            if isdf_to_save is not None:
                self._isdf = isdf_to_save
                dump(isdf_to_save, "inpv_kpt", inpv_kpt)
                dump(isdf_to_save, "coul_kpt", coul_kpt)
                nbytes = inpv_kpt.nbytes + coul_kpt.nbytes
                log.info("ISDF results are saved to %s, size = %6.2e GB", isdf_to_save, nbytes / 1e9)

        comm.barrier()

    def build_eta_kpt(self, inpv_kpt):
        comm = self._comm
        comm.barrier()

        rank = comm.Get_rank()
        size = comm.Get_size()

        log = logger.new_logger(self, self.verbose)
        max_memory = max(2000, self.max_memory - current_memory()[0]) # in MB

        cell = self.cell
        kpts = self.kpts
        phase = get_phase_factor(cell, kpts)

        grids = self.grids
        ngrid = grids.coords.shape[0]
        nkpt, nip, nao = inpv_kpt.shape

        blksize_max = int(max_memory * 1e6 * 0.2) // (nkpt * nip * 16)
        blksize_max = max(BLKSIZE, blksize_max)
        blksize_max = min(CONTRACT_MAX_SIZE, blksize_max)
        blksize = compute_blksize(ngrid, blksize_max, BLKSIZE)
        
        shape = (nkpt * nip, ngrid)
        dtype = numpy.complex128

        log.debug("\nMPI version is used for eta_kpt")
        log.debug("shape = %s", shape)
        log.debug("disk space required: %6.2e GB", numpy.prod(shape) * 16 / 1e9)
        log.debug("blksize = %d, ngrid = %d", blksize, ngrid)
        log.debug("approximate memory needed for each block:   %6.2e GB", nkpt * nip * 16 * blksize / 1e9)
        log.debug("approximate memory needed for each k-point: %6.2e GB", nip * ngrid * 16 / 1e9)
        log.debug("max_memory: %6.2e GB", max_memory / 1e3)
        comm.barrier()

        fswap = self._fswap
        eta_kpt = fswap.create_dataset("eta_kpt", shape=shape, dtype=dtype)
        comm.barrier()

        log.debug("\nComputing eta_kpt")
        info = (lambda s: f"eta_kpt[ %{len(s)}d: %{len(s)}d]")(str(ngrid))
        block_loop = self.gen_block_loop(blksize=blksize)
        for iblock, (ao_etc_kpt, g0, g1) in enumerate(block_loop):
            t0 = (process_clock(), perf_counter())
            if iblock % size != rank:
                continue
            ao_kpt = numpy.asarray(ao_etc_kpt[0], dtype=numpy.complex128)

            eta_kpt_g0g1 = contract(inpv_kpt, ao_kpt, phase)
            eta_kpt_g0g1 = eta_kpt_g0g1.reshape(nkpt * nip, g1 - g0)

            eta_kpt[:, g0:g1] = eta_kpt_g0g1
            eta_kpt_g0g1 = None

            log.timer(info % (g0, g1), *t0)

        comm.barrier()
        return eta_kpt

    def build_coul_kpt(self, inpv_kpt, eta_kpt):
        comm = self._comm
        comm.barrier()

        rank = comm.Get_rank()
        size = comm.Get_size()

        log = logger.new_logger(self, self.verbose)
        tol = self.tol

        cell = self.cell
        kpts = self.kpts
        phase = get_phase_factor(cell, kpts)

        mesh = cell.mesh
        v0 = cell.get_Gv(mesh)
        
        grids = self.grids
        coord = grids.coords
        ngrid = coord.shape[0]

        nkpt, nip, nao = inpv_kpt.shape
        metx_kpt = contract(inpv_kpt, inpv_kpt, phase)
        comm.barrier()

        fswap = self._fswap
        shape = (nkpt, nip, nip)
        chunk = (1, nip, nip)
        dtype = numpy.complex128
        coul_kpt = fswap.create_dataset("coul_kpt", shape=shape, dtype=dtype, chunks=chunk)
        comm.barrier()

        log.debug("\nComputing coul_kpt")
        info = (lambda s: f"coul_kpt[ %{len(s)}d / {s}]")(str(nkpt))
        for q in range(nkpt):
            t0 = (process_clock(), perf_counter())
            q0, q1 = q * nip, (q + 1) * nip
            if q % size != rank:
                continue
            
            fq = numpy.exp(-1j * coord @ kpts[q])
            vq = pbctools.get_coulG(cell, k=kpts[q], exx=False, Gv=v0, mesh=mesh)
            vq *= cell.vol / ngrid
            lq = eta_kpt[q0:q1, :] * fq

            wq = pbctools.fft(lq, mesh)
            rq = pbctools.ifft(wq * vq, mesh)
            rq = rq.conj()

            kern_q = lib.dot(lq, rq.T) / numpy.sqrt(ngrid)
            lq = rq = wq = None

            metx_q = metx_kpt[q]
            res = lstsq(metx_q, kern_q, tol=tol)
            coul_q = res[0]
            coul_q = (coul_q + coul_q.conj().T) / 2
            if log.verbose >= logger.DEBUG1:
                err = metx_q @ coul_q @ metx_q - kern_q
                err = abs(err).max() / abs(kern_q).max()
                log.debug("\nMetric tensor rank: %d / %d, lstsq error: %6.2e", res[1], nip, err)
                
            coul_kpt[q] = coul_q * numpy.sqrt(ngrid)
            coul_q = None

            log.timer(info % (q + 1), *t0)

        comm.barrier()
        coul_kpt = fswap["coul_kpt"][:]
        comm.barrier()

        return coul_kpt

FFTISDF = ISDF = WithMPI
