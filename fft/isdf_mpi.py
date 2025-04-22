import mpi4py
from mpi4py import MPI

import os, sys, h5py
import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import lib
from pyscf.lib import logger, current_memory
from pyscf.lib.logger import process_clock, perf_counter

from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero

from pyscf.pbc.tools.k2gamma import get_phase
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 2000))

import df.fftisdf as fftisdf
from df.fftisdf import kpts_to_kmesh
from df.fftisdf import spc_to_kpt, kpt_to_spc
# from fft_isdf import get_lhs_and_rhs, get_kern

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def build(df_obj, inpx=None):
    """
    Build the FFT-ISDF object with proper MPI support.
    """
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())

    cell = df_obj.cell
    wrap_around = df_obj.wrap_around
    kpts, kmesh = kpts_to_kmesh(cell, df_obj.kpts, wrap_around)
    phase = get_phase(cell, kpts, kmesh, wrap_around)[1]
    nkpt = len(kpts)

    nip = inpx.shape[0]
    assert inpx.shape == (nip, 3)
    nao = cell.nao_nr()

    from h5py import File
    from tempfile import NamedTemporaryFile
    _tmpfile = NamedTemporaryFile(dir=lib.param.TMPDIR)
    tmpfile  = _tmpfile.name
    
    # Only root process creates and writes to the file
    if rank == 0:
        # Calculate data on root process
        inpv_kpt = cell.pbc_eval_gto("GTOval", inpx, kpts=kpts)
        inpv_kpt = numpy.asarray(inpv_kpt, dtype=numpy.complex128)
        assert inpv_kpt.shape == (nkpt, nip, nao)
        log.debug("nip = %d, nao = %d, cisdf = %6.2f", nip, nao, nip / nao)
        
        ngrid = df_obj.grids.coords.shape[0]
        max_memory = max(2000, df_obj.max_memory - current_memory()[0])
        
        # Generate metx_kpt and eta_kpt data, must use 
        # out-of-core version to pass data among processes
        from df.fftisdf import get_lhs_and_rhs
        df_obj.fswap = File(tmpfile, 'w')
        metx_kpt, eta_kpt = get_lhs_and_rhs(df_obj, inpv_kpt, max_memory=max_memory)
        df_obj.fswap.create_dataset('inpv_kpt', data=inpv_kpt)
        df_obj.fswap.create_dataset('metx_kpt', data=metx_kpt)
        df_obj.fswap.create_dataset('eta_kpt',  data=eta_kpt)
        df_obj.fswap.create_dataset('coul_kpt', data=numpy.zeros((nkpt, nip, nip), dtype=numpy.complex128))
        df_obj.fswap.close()
        df_obj.fswap = None

    comm.barrier()
    
    # Broadcast the file name to all processes
    tmpfile = comm.bcast(tmpfile, root=0)
    
    fswap = File(tmpfile, "r+", driver="mpio", comm=comm)
    metx_kpt = fswap['metx_kpt'][:]
    inpv_kpt = fswap['inpv_kpt'][:]

    for q in range(nkpt):
        if q % size != rank:
            continue
            
        metx_q = metx_kpt[q]
        
        from df.fftisdf import get_kern, lstsq
        eta_q = fswap['eta_kpt'][q]
        kern_q = get_kern(df_obj, eta_q=eta_q, kpt_q=kpts[q])
        coul_q = lstsq(metx_q, kern_q, tol=df_obj.tol, verbose=df_obj)
        fswap['coul_kpt'][q] = coul_q

        log.info("Finished solving Coulomb kernel for q = %3d / %3d, rank = %d / %d", q + 1, nkpt, rank, size)
        log.stdout.flush()

    coul_kpt = fswap['coul_kpt'][:]
    fswap.close()
    comm.barrier()
    return inpv_kpt, coul_kpt

fftisdf.build = build

class WithMPI(fftisdf.FFTISDF):
    def build_eta_kpt(self, inpv_kpt):
        log = logger.new_logger(self, self.verbose)
        max_memory = max(2000, self.max_memory - current_memory()[0]) # in MB

        cell = self.cell
        wrap_around = self.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, self.kpts, wrap_around)
        phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=wrap_around)[1]

        grids = self.grids
        ngrid = grids.coords.shape[0]
        nkpt, nip, nao = inpv_kpt.shape

        blksize = self.blksize
        if blksize is None:
            blksize = max_memory * 1e6 * 0.2
            blksize = int(blksize) // (nkpt * nip * 16)
            blksize = max(blksize, 1)
        blksize = min(blksize, int(ngrid / size))

        if rank == 0:
            self._fswap = lib.H5TmpFile()
            fswap.close()

        comm.barrier()
        fswap = comm.bcast(self._fswap.name, root=0)





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
