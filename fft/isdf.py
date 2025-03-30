import os, sys, h5py
from functools import reduce

import numpy, scipy
import scipy.linalg
from scipy.linalg import svd

import pyscf

# pyscf.lib
from pyscf import lib
from pyscf.lib import logger, current_memory
from pyscf.lib.chkfile import load, dump
from pyscf.lib.logger import process_clock, perf_counter

# pyscf.pbc
from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools.pbc import fft, ifft
from pyscf.pbc.tools.k2gamma import get_phase
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

# fft.isdf_jk
from fft.isdf_jk import get_k_kpts, kpts_to_kmesh, spc_to_kpt, kpt_to_spc

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 2000))

# Naming convention:
# *_kpt: k-space array, which shapes as (nkpt, x, x)
# *_spc: super-cell stripe array, which shapes as (nspc, x, x)
# *_full: full array, shapes as (nspc * x, nspc * x)
# *_k1, *_k2: the k-space array at specified k-point

def _eta_kpt_g0g1(phi, inpv_kpt, phase):
    """Compute the Coulomb kernel in k-space for a given grid range."""
    nkpt, ngrid, nao = phi.shape
    nip = inpv_kpt.shape[1]

    t_kpt = numpy.asarray([fk.conj() @ xk.T for fk, xk in zip(phi, inpv_kpt)])
    assert t_kpt.shape == (nkpt, ngrid, nip)
    t_spc = kpt_to_spc(t_kpt, phase)
    
    eta_spc_g0g1 = t_spc ** 2
    eta_kpt_g0g1 = spc_to_kpt(eta_spc_g0g1, phase).conj()

    return eta_kpt_g0g1

def _coul_q(metx_q, eta_q, vq=None, fq=None, mesh=None, tol=1e-10):
    """Compute the Coulomb kernel in k-space for a given grid range."""
    ngrid, nip = eta_q.shape
    v_q = fft(eta_q.T * fq, mesh) * vq
    w_q = ifft(v_q, mesh) * fq.conj()
    assert w_q.shape == (nip, ngrid)

    kern_q = lib.dot(w_q, eta_q.conj())
    assert kern_q.shape == (nip, nip)

    u, s, vh = svd(metx_q, full_matrices=False)
    s2 = s[:, None] * s[None, :]
    m = abs(s2) > tol ** 2
    s2inv = numpy.zeros_like(s2)
    s2inv[m] = 1.0 / s2[m]

    coul_q = reduce(lib.dot, (u.conj().T, kern_q, u))
    coul_q *= s2inv
    coul_q = reduce(lib.dot, (vh.conj().T, coul_q, vh))
    return coul_q

def build(df_obj, inpx=None, verbose=0):
    """
    Build the FFT-ISDF object.
    
    Args:
        df_obj: The FFT-ISDF object to build.
    """
    log = logger.new_logger(df_obj, verbose)
    t0 = (process_clock(), perf_counter())
    max_memory = max(2000, df_obj.max_memory - current_memory()[0])

    if df_obj._isdf is not None:
        log.info("Loading ISDF results from %s, skipping build", df_obj._isdf)
        inpv_kpt = load(df_obj._isdf, "inpv_kpt")
        coul_kpt = load(df_obj._isdf, "coul_kpt")
        df_obj._inpv_kpt = inpv_kpt
        df_obj._coul_kpt = coul_kpt
        return inpv_kpt, coul_kpt

    df_obj.dump_flags()
    df_obj.check_sanity()

    cell = df_obj.cell
    nao = cell.nao_nr()
    kpts, kmesh = kpts_to_kmesh(df_obj, df_obj.kpts)
    nkpt = len(kpts)

    if inpx is None:
        inpx = df_obj.get_inpx(g0=None, c0=df_obj.c0)

    nip = inpx.shape[0]
    assert inpx.shape == (nip, 3)

    grids = df_obj.grids
    ngrid = grids.coords.shape[0]

    if df_obj.blksize is None:
        blksize = max_memory * 1e6 * 0.2 / (nkpt * nip * 16)
        df_obj.blksize = max(1, int(blksize))
    df_obj.blksize = min(df_obj.blksize, ngrid)

    if df_obj.blksize >= ngrid:
        df_obj._fswap = None

    inpv_kpt = cell.pbc_eval_gto("GTOval", inpx, kpts=kpts)
    inpv_kpt = numpy.asarray(inpv_kpt, dtype=numpy.complex128)
    assert inpv_kpt.shape == (nkpt, nip, nao)
    log.debug("nip = %d, nao = %d, cisdf = %6.2f", nip, nao, nip / nao)
    t1 = log.timer("get interpolating vectors")
    
    fswap = None if df_obj._fswap is None else h5py.File(df_obj._fswap, "w")
    if fswap is None:
        log.debug("In-core version is used for eta_kpt, memory required = %6.2e GB, max_memory = %6.2e GB", nkpt * nip * 16 * ngrid / 1e9, max_memory / 1e3)
    else:
        log.debug("Out-core version is used for eta_kpt, disk space required = %6.2e GB.", nkpt * nip * 16 * ngrid / 1e9)
        log.debug("memory used for each block = %6.2e GB, each k-point = %6.2e GB", nkpt * nip * 16 * df_obj.blksize / 1e9, nip * ngrid * 16 / 1e9)
        log.debug("max_memory = %6.2e GB", max_memory / 1e3)

    # metx_kpt: (nkpt, nip, nip), eta_kpt: (nkpt, ngrid, nip)
    # assume metx_kpt is a numpy.array, while eta_kpt is a hdf5 dataset
    # metx_kpt, eta_kpt = get_lhs_and_rhs(df_obj, inpv_kpt, fswap=fswap)
    phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=df_obj.wrap_around)[1]

    # compute metrix tensor in k-space
    t_kpt = numpy.asarray([xk.conj() @ xk.T for xk in inpv_kpt])
    t_spc = kpt_to_spc(t_kpt, phase)
    metx_kpt = spc_to_kpt(t_spc * t_spc, phase)

    # compute Coulomb kernel in k-space
    eta_kpt = fswap.create_dataset("eta_kpt", shape=(nkpt, ngrid, nip), dtype=numpy.complex128) \
        if fswap is not None else numpy.zeros((nkpt, ngrid, nip), dtype=numpy.complex128)
    
    aoR_loop = df_obj.aoR_loop(grids, kpts, 0, blksize=df_obj.blksize)
    for ig, (ao_kpt, g0, g1) in enumerate(aoR_loop):
        phi = numpy.asarray(ao_kpt[0])
        eta_kpt[:, g0:g1, :] = _eta_kpt_g0g1(phi, inpv_kpt, phase)

    coul_kpt = []
    for q in range(nkpt):
        metx_q = metx_kpt[q]
        eta_q = eta_kpt[q]

        vq = pbctools.get_coulG(cell, k=kpts[q], mesh=cell.mesh) * cell.vol / ngrid

        tq = numpy.dot(grids.coords, kpts[q])
        fq = numpy.exp(-1j * tq) # .reshape(ngrid, 1)

        coul_q = _coul_q(metx_q, eta_q, vq, fq, mesh=cell.mesh, tol=1e-10)
        coul_kpt.append(coul_q)

    coul_kpt = numpy.asarray(coul_kpt)
    df_obj._coul_kpt = coul_kpt
    df_obj._inpv_kpt = inpv_kpt

    if df_obj._isdf_to_save is not None:
        df_obj._isdf = df_obj._isdf_to_save

    if df_obj._isdf is not None:
        dump(df_obj._isdf, "coul_kpt", coul_kpt)
        dump(df_obj._isdf, "inpv_kpt", inpv_kpt)

    t1 = log.timer("building ISDF", *t0)
    if fswap is not None:
        fswap.close()
    return inpv_kpt, coul_kpt

class InterpolativeSeparableDensityFitting(FFTDF):
    wrap_around = False

    _fswap = None
    _isdf = None
    _isdf_to_save = None

    _coul_kpt = None
    _inpv_kpt = None

    blksize = None
    tol = 1e-10
    c0 = 10.0

    _keys = {"tol", "c0"}

    def __init__(self, cell, kpts=numpy.zeros((1, 3))):
        FFTDF.__init__(self, cell, kpts)
        from tempfile import NamedTemporaryFile
        fswap = NamedTemporaryFile(dir=lib.param.TMPDIR)
        self._fswap = fswap.name

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info("mesh = %s (%d PWs)", self.mesh, numpy.prod(self.mesh))
        log.info("len(kpts) = %d", len(self.kpts))
        return self
    
    build = build
    
    def aoR_loop(self, grids=None, kpts=None, deriv=0, blksize=None):
        if grids is None:
            grids = self.grids
            cell = self.cell
        else:
            cell = grids.cell

        if grids.non0tab is None:
            grids.build(with_non0tab=True)

        if kpts is None:
            kpts = self.kpts
        kpts = numpy.asarray(kpts)

        assert cell.dimension == 3

        max_memory = max(2000, self.max_memory - current_memory()[0])

        ni = self._numint
        nao = cell.nao_nr()
        p1 = 0

        block_loop = ni.block_loop(
            cell, grids, nao, deriv, kpts,
            max_memory=max_memory,
            blksize=blksize
            )
        
        for ao_etc_kpt in block_loop:
            coords = ao_etc_kpt[4]
            p0, p1 = p1, p1 + coords.shape[0]
            yield ao_etc_kpt, p0, p1
    
    def get_inpx(self, g0=None, c0=None, tol=None):
        log = logger.new_logger(self, self.verbose)
        t0 = (process_clock(), perf_counter())

        if g0 is None:
            assert c0 is not None
            nip = self.cell.nao_nr() * c0

            from pyscf.pbc.tools.pbc import mesh_to_cutoff
            lv = self.cell.lattice_vectors()
            k0 = mesh_to_cutoff(lv, [int(numpy.power(nip, 1/3) + 1)] * 3)
            k0 = max(k0)

            from pyscf.pbc.tools.pbc import cutoff_to_mesh
            g0 = self.cell.gen_uniform_grids(cutoff_to_mesh(lv, k0))

        if tol is None:
            tol = self.tol
        
        pcell = self.cell
        ng = len(g0)

        x0 = pcell.pbc_eval_gto("GTOval", g0)
        m0 = lib.dot(x0.conj(), x0.T) ** 2

        from pyscf.lib.scipy_helper import pivoted_cholesky
        tol2 = tol ** 2
        chol, perm, rank = pivoted_cholesky(m0, tol=tol2)

        nip = pcell.nao_nr() * c0 if c0 is not None else rank
        nip = int(nip)
        mask = perm[:nip]

        nip = mask.shape[0]
        log.info("Pivoted Cholesky rank = %d, estimated error = %6.2e", rank, chol[nip-1, nip-1])
        log.info("Parent grid size = %d, selected grid size = %d", ng, nip)

        inpx = g0[mask]
        t1 = log.timer("interpolating functions", *t0)
        return inpx
    
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        
        assert omega is None and exxdiv is None
        from pyscf.pbc.df.aft import _check_kpts
        kpts, is_single_kpt = _check_kpts(self, kpts)

        vj = vk = None
        if with_k:
            vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            from pyscf.pbc.df.fft_jk import get_j_kpts
            vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)

        return vj, vk

ISDF = FFTISDF = InterpolativeSeparableDensityFitting

if __name__ == "__main__":
    cell = pyscf.pbc.gto.M(
    a = numpy.ones((3, 3)) * 3.5668 - numpy.eye(3) * 3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917''',
    basis = 'gth-dzvp', pseudo = 'gth-pade',
    verbose = 4, ke_cutoff = 20.0
    )

    nao = cell.nao_nr()
    kmesh = [4, 4, 4]
    nkpt = nspc = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    scf_obj.conv_tol = 1e-8
    dm_kpts = scf_obj.get_init_guess(key="minao")

    log = logger.new_logger(None, 5)

    vv = []
    ee = []
    cc = [5.0, 10.0, 15.0, 20.0]
    for c0 in cc:
        from pyscf.pbc.tools.pbc import cutoff_to_mesh
        lv = cell.lattice_vectors()
        g0 = cell.gen_uniform_grids(cutoff_to_mesh(lv, cell.ke_cutoff))

        from fft import FFTISDF
        scf_obj.with_df = FFTISDF(cell, kpts=kpts)
        scf_obj.with_df.verbose = 10
        scf_obj.with_df.tol = 1e-10
        scf_obj.with_df.max_memory = 2000

        df_obj = scf_obj.with_df
        inpx = df_obj.get_inpx(g0=g0, c0=c0, tol=1e-10)
        df_obj.build(inpx=inpx, verbose=10)

        vj, vk = df_obj.get_jk(dm_kpts)
        vv.append((vj, vk))
        ee.append(scf_obj.energy_tot(dm_kpts))

    from pyscf.pbc.df.fft import FFTDF
    scf_obj.with_df = FFTDF(cell, kpts)
    scf_obj.with_df.verbose = 0
    vj_ref, vk_ref = scf_obj.with_df.get_jk(dm_kpts)
    e_ref = scf_obj.energy_tot(dm_kpts)

    print("-> FFTDF e_tot = %12.8f" % e_ref)
    for ic, c0 in enumerate(cc):
        print("-> FFTISDF c0 = %6s, ene_err = % 6.2e, vj_err = % 6.2e, vk_err = % 6.2e" % (c0, abs(ee[ic] - e_ref), abs(vv[ic][0] - vj_ref).max(), abs(vv[ic][1] - vk_ref).max()))

