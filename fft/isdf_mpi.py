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

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 2000))

class WithMPI(fft.isdf.ISDF):
    comm = None
    def __init__(self, cell, kpts, comm=None):
        super().__init__(cell, kpts)
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self._comm = comm
        comm.barrier()

        rank = comm.Get_rank()
        if rank != 0:
            self._tmp = None
            self._tmpfile = None
        self._tmpfile = self._comm.bcast(self._tmpfile, root=0)
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
        if self.blksize is not None:
            log.warning("In MPI version, blksize is not used")

        blksize_max = int(max_memory * 1e6 * 0.2) // (nkpt * nip * 16)
        blksize_max = max(blksize_max, 1)
        blksize_max = min(blksize_max, ngrid // size + 1)
        blksize = ngrid // (ngrid // blksize_max + 1) + 1

        log.debug("\nMPI version is used for eta_kpt")
        log.debug("disk space required: %6.2e GB", nkpt * nip * 16 * ngrid / 1e9)
        log.debug("blksize = %d, ngrid = %d", blksize, ngrid)
        log.debug("memory needed for each block:   %6.2e GB", nkpt * nip * 16 * blksize / 1e9)
        log.debug("memory needed for each k-point: %6.2e GB", nip * ngrid * 16 / 1e9)
        log.debug("max_memory: %6.2e GB", max_memory / 1e3)

        from h5py import File
        fswap = comm.bcast(self._tmpfile, root=0)
        comm.barrier()

        self._fswap = File(fswap, "w", driver="mpio", comm=comm)
        eta_kpt = self._fswap.create_dataset("eta_kpt", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
        comm.barrier()

        aoR_loop = self.aoR_loop(grids, kpts, 0, blksize=blksize)
        log.debug("\nComputing eta_kpt")
        info = (lambda s: f"eta_kpt[ %{len(s)}d: %{len(s)}d]")(str(ngrid))
        for ig, (ao_etc_kpt, g0, g1) in enumerate(aoR_loop):
            t0 = (process_clock(), perf_counter())
            if ig % size != rank:
                continue
            ao_kpt = numpy.asarray(ao_etc_kpt[0])
            eta_kpt_g0g1 = contract(inpv_kpt, ao_kpt, phase)
            eta_kpt_g0g1 = eta_kpt_g0g1.transpose(0, 2, 1)
            eta_kpt[:, g0:g1, :] = eta_kpt_g0g1
            eta_kpt_g0g1 = None

            log.timer(info % (g0, g1), *t0)
        comm.barrier()

        print("eta_kpt.shape = %s" % str(eta_kpt.shape), "rank = %d" % rank)
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
        coul_kpt = self._fswap.create_dataset("coul_kpt", shape=(nkpt, nip, nip), dtype=numpy.complex128)

        log.debug("\nComputing coul_kpt")
        info = (lambda s: f"coul_kpt[ %{len(s)}d / {s}]")(str(nkpt))
        for q in range(nkpt):
            if q % size != rank:
                continue

            t0 = (process_clock(), perf_counter())
            
            fq = numpy.exp(-1j * coords @ kpts[q])
            vq = pbctools.get_coulG(cell, k=kpts[q], Gv=v0, mesh=mesh)
            vq *= cell.vol / ngrid
            
            from pyscf.pbc.tools.pbc import fft, ifft
            lq = eta_kpt[q].T * fq
            wq = fft(lq, mesh)
            rq = ifft(wq * vq, mesh)
            kern_q = lib.dot(lq, rq.conj().T)

            metx_q = metx_kpt[q]
            coul_q = lstsq(metx_q, kern_q, tol=tol2)

            coul_kpt[q] = coul_q
            log.timer(info % (q + 1), *t0)
        comm.barrier()

        log.debug("Broadcasting coul_kpt")
        coul_kpt = self._fswap["coul_kpt"][:]
        print("coul_kpt.shape = %s" % str(coul_kpt.shape), "rank = %d" % rank)
        return coul_kpt

FFTISDF = ISDF = WithMPI

if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", "../data/")
    from utils import cell_from_poscar

    TMPDIR = lib.param.TMPDIR

    cell = cell_from_poscar(os.path.join(DATA_PATH, "diamond-prim.vasp"))
    cell.basis = 'gth-dzvp-molopt-sr'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.exp_to_discard = 0.1
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.ke_cutoff = 100.0
    cell.stdout = sys.stdout if rank == 0 else open(TMPDIR + "out-%d.log" % rank, "w")
    cell.build(dump_input=False)

    nao = cell.nao_nr()

    # kmesh = [4, 4, 4]
    kmesh = [4, 4, 4]
    nkpt = nspc = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    scf_obj.stdout = cell.stdout
    dm_kpts = scf_obj.get_init_guess(key="minao")

    log = logger.new_logger(cell, 5)
    log.stdout = cell.stdout

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = FFTDF(cell, kpts)
    scf_obj.with_df.verbose = 5
    scf_obj.with_df.stdout = cell.stdout

    if rank == 0:
        scf_obj.with_df.dump_flags()
        scf_obj.with_df.check_sanity()

    vj0 = numpy.zeros((nkpt, nao, nao))
    vk0 = numpy.zeros((nkpt, nao, nao))
    # vj0, vk0 = scf_obj.get_jk(dm_kpts=dm_kpts, with_j=True, with_k=True)
    vj0 = vj0.reshape(nkpt, nao, nao)
    vk0 = vk0.reshape(nkpt, nao, nao)

    c0 = 5.0
    from pyscf.pbc.tools.pbc import cutoff_to_mesh
    lv = cell.lattice_vectors()
    g0 = cell.gen_uniform_grids(cutoff_to_mesh(lv, cell.ke_cutoff))

    scf_obj.with_df = ISDF(cell, kpts=kpts)
    scf_obj.with_df.verbose = 5
    scf_obj.with_df.tol = 1e-10
    scf_obj.with_df.max_memory = 2000

    df_obj = scf_obj.with_df
    inpx = df_obj.get_inpx(g0=g0, c0=c0, tol=1e-10)
    df_obj.build(inpx)

    vj1, vk1 = df_obj.get_jk(dm_kpts)
    vj1 = vj1.reshape(nkpt, nao, nao)
    vk1 = vk1.reshape(nkpt, nao, nao)

    if rank == 0:
        err_j = abs(vj0 - vj1).max()
        err_k = abs(vk0 - vk1).max()
        print("err_j = %12.6e, err_k = %12.6e" % (err_j, err_k))
