import fft.isdf

import os, sys, h5py
import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import lib
from pyscf.lib import logger, current_memory
from pyscf.lib.chkfile import dump
from pyscf.lib.logger import process_clock, perf_counter

from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools.k2gamma import get_phase

import fft
from fft.isdf import contract, lstsq
from fft.isdf import kpts_to_kmesh

from pyscf import __config__
PARENT_GRID_SIZE_MAX = getattr(__config__, "isdf_parent_grid_size_max", 40000)
CONTRACT_BLKSIZE_MAX = getattr(__config__, "isdf_contract_blksize_max", 40000)

class WithMPI(fft.isdf.ISDF):
    comm = None
    def __init__(self, cell, kpts, comm=None):
        super().__init__(cell, kpts)
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self._comm = comm
        comm.barrier()

        fswap = self._comm.bcast(self._fswap.filename, root=0)
        self._fswap.close()

        self._fswap = lib.H5TmpFile(fswap, "w", driver="mpio", comm=comm)
        comm.barrier()

    def save(self, isdf_to_save=None):
        log = logger.new_logger(self, self.verbose)
        comm = self._comm
        comm.barrier()

        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            inpv_kpt = self._inpv_kpt
            coul_kpt = self._coul_kpt
            assert inpv_kpt is not None
            assert coul_kpt is not None

            dump(isdf_to_save, "inpv_kpt", inpv_kpt)
            dump(isdf_to_save, "coul_kpt", coul_kpt)

            nbytes = inpv_kpt.nbytes + coul_kpt.nbytes
            log.info("ISDF results are saved to %s, size = %d MB", isdf_to_save, nbytes / 1e6)

        comm.barrier()

    def build_eta_kpt(self, inpv_kpt):
        log = logger.new_logger(self, self.verbose)
        comm = self._comm
        comm.barrier()

        rank = comm.Get_rank()
        size = comm.Get_size()
        max_memory = max(2000, self.max_memory - current_memory()[0]) # in MB

        cell = self.cell
        wrap_around = self.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, self.kpts, wrap_around)
        phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=wrap_around)[1]

        grids = self.grids
        ngrid = grids.coords.shape[0]
        nkpt, nip, nao = inpv_kpt.shape

        blksize = int(max_memory * 1e6 * 0.1) // (nkpt * nip * 16)
        blksize = max(blksize, 1)
        blksize = min(CONTRACT_BLKSIZE_MAX, blksize, ngrid)
        blknum = (ngrid + blksize - 1) // blksize
        blknum = (blknum + size - 1) // size * size
        blksize = ngrid // blknum + 1

        log.debug("\nMPI version is used for eta_kpt")
        log.debug("disk space required: %6.2e GB", nkpt * nip * 16 * ngrid / 1e9)
        log.debug("blksize = %d, ngrid = %d", blksize, ngrid)
        log.debug("memory needed for each block:   %6.2e GB", nkpt * nip * 16 * blksize / 1e9)
        log.debug("memory needed for each k-point: %6.2e GB", nip * ngrid * 16 / 1e9)
        log.debug("max_memory: %6.2e GB", max_memory / 1e3)

        fswap = self._fswap
        eta_kpt = fswap.create_dataset("eta_kpt", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
        comm.barrier()

        aoR_loop = self.aoR_loop(grids, kpts, 0, blksize=blksize)
        log.debug("\nComputing eta_kpt")
        info = (lambda s: f"eta_kpt[ %{len(s)}d: %{len(s)}d]")(str(ngrid))
        for ig, (ao_etc_kpt, g0, g1) in enumerate(aoR_loop):
            t0 = (process_clock(), perf_counter())
            if ig % size != rank:
                continue
            ao_kpt = numpy.asarray(ao_etc_kpt[0], dtype=numpy.complex128)

            # eta_kpt_g0g1: (nkpt, nip, g1 - g0)
            eta_kpt_g0g1 = contract(inpv_kpt, ao_kpt, phase)
            eta_kpt_g0g1 = eta_kpt_g0g1.transpose(0, 2, 1)

            eta_kpt[:, g0:g1, :] = eta_kpt_g0g1
            eta_kpt_g0g1 = None

            log.timer(info % (g0, g1), *t0)

        comm.barrier()
        return eta_kpt

    def build_coul_kpt(self, inpv_kpt, eta_kpt):
        log = logger.new_logger(self, self.verbose)
        comm = self._comm
        comm.barrier()

        rank = comm.Get_rank()
        size = comm.Get_size()
        tol2 = self.tol ** 2

        cell = self.cell
        wrap_around = self.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, self.kpts, wrap_around)
        phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=wrap_around)[1]
        mesh = cell.mesh
        v0 = cell.get_Gv(mesh)
        
        grids = self.grids
        coords = grids.coords
        ngrid = coords.shape[0]

        nkpt, nip, nao = inpv_kpt.shape
        metx_kpt = contract(inpv_kpt, inpv_kpt, phase)

        fswap = self._fswap
        coul_kpt = fswap.create_dataset("coul_kpt", shape=(nkpt, nip, nip), dtype=numpy.complex128)

        log.debug("\nComputing coul_kpt")
        info = (lambda s: f"coul_kpt[ %{len(s)}d / {s}]")(str(nkpt))
        for q in range(nkpt):
            if q % size != rank:
                continue

            t0 = (process_clock(), perf_counter())
            
            fq = numpy.exp(-1j * coords @ kpts[q])
            vq = pbctools.get_coulG(cell, k=kpts[q], Gv=v0, mesh=mesh)
            vq *= cell.vol / ngrid
            lq = eta_kpt[q].T * fq
            wq = pbctools.fft(lq, mesh)
            rq = pbctools.ifft(wq * vq, mesh)
            kern_q = lq @ rq.conj().T
            lq = rq = wq = None

            metx_q = metx_kpt[q]
            coul_q = lstsq(metx_q, kern_q, tol=tol2)
            coul_kpt[q] = coul_q
            log.timer(info % (q + 1), *t0)

        comm.barrier()

        log.debug("Broadcasting coul_kpt")
        coul_kpt = self._fswap["coul_kpt"][:]
        comm.barrier()

        return coul_kpt

FFTISDF = ISDF = WithMPI
