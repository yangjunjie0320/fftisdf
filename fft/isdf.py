import os, sys, h5py
from functools import reduce

import numpy, scipy
import scipy.linalg
from scipy.linalg import svd

import pyscf
from pyscf import __config__

# pyscf.lib
from pyscf import lib
from pyscf.lib import logger, current_memory
from pyscf.lib.chkfile import load, dump
from pyscf.lib.logger import process_clock, perf_counter
from pyscf.lib.scipy_helper import pivoted_cholesky

# pyscf.pbc
from pyscf.pbc import tools as pbctools
from pyscf.pbc.df.aft import _check_kpts
from pyscf.pbc.df.fft import FFTDF

from fft.isdf_jk import get_k_kpts
from fft.isdf_jk import get_phase, kpts_to_kmesh
from fft.isdf_jk import spc_to_kpt, kpt_to_spc

PARENT_GRID_SIZE_MAX = getattr(__config__, "isdf_parent_grid_size_max", 40000)
CONTRACT_BLKSIZE_MAX = getattr(__config__, "isdf_contract_blksize_max", 40000)

# Naming convention:
# *_kpt: k-space array, which shapes as (nkpt, x, x)
# *_spc: super-cell stripe array, which shapes as (nspc, x, x)
# *_full: full array, shapes as (nspc * x, nspc * x)
# *_k1, *_k2: the k-space array at specified k-point

def contract(f_kpt, g_kpt, phase):
    """Contract two k-space arrays."""
    nk, m, n = f_kpt.shape
    l = g_kpt.shape[1]
    assert f_kpt.shape == (nk, m, n)
    assert g_kpt.shape == (nk, l, n)

    # kmn,kln -> kml
    t_kpt = f_kpt.conj() @ g_kpt.transpose(0, 2, 1)
    t_kpt = t_kpt.reshape(nk, m, l)
    
    t_spc = kpt_to_spc(t_kpt, phase)

    # smn,smn -> smn
    x_spc = t_spc * t_spc
    x_kpt = spc_to_kpt(x_spc, phase)
    return x_kpt

def lstsq(a, b, tol=1e-10):
    u, s, vh = svd(a, full_matrices=False)
    uh = u.conj().T
    v = vh.conj().T
    s2 = s[None, :] * s[:, None]
    
    mask = abs(s2) > tol
    x = reduce(numpy.dot, (uh, b, u))
    x[mask] /= s2[mask]
    x = reduce(numpy.dot, (v, x, vh))

    x = (x + x.conj().T) / 2
    return x

def select_inpx(df_obj, g0=None, c0=None, kpts=None, tol=1e-10):
    log = logger.new_logger(df_obj, df_obj.verbose)
    
    cell = df_obj.cell
    nao = cell.nao_nr()
    ng = g0.shape[0]

    wrap_around = df_obj.wrap_around
    kpts, kmesh = kpts_to_kmesh(cell, kpts, wrap_around)
    phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=wrap_around)[1]
    nkpt = len(kpts)

    if kpts is None:
        kpts = numpy.zeros((1, 3))

    x_kpt = cell.pbc_eval_gto("GTOval", g0, kpts=kpts)
    x_kpt = numpy.asarray(x_kpt, dtype=numpy.complex128)
    t = x_kpt.transpose(1, 0, 2).reshape(ng, -1)
    m0 = t.conj() @ t.T / numpy.sqrt(nkpt)
    m0 = m0.real ** 2

    chol, perm, rank = pivoted_cholesky(m0, tol=tol)
    nip = rank

    if c0 is not None:
        nip = int(c0 * nao)
        log.info("Cholesky rank = %d, c0 = %6.2f, nao = %d, nip = %d", rank, c0, nao, nip)
    else:
        log.info("Cholesky rank = %d, nao = %d, nip = %d", rank, nao, nip)
        log.info("Using all Cholesky vectors as interpolating points.")

    mask = perm[:nip]
    diag = numpy.diag(chol)
    s0 = numpy.sum(diag[:nip])
    s1 = numpy.sum(diag[nip:])

    log.info("Parent grid size = %d, selected grid size = %d", ng, nip)
    log.info("truncated values = %6.2e, estimated error = %6.2e", s0, s1)

    inpx = g0[mask]
    return inpx
    
class InterpolativeSeparableDensityFitting(FFTDF):
    wrap_around = False
    tol = 1e-8
    c0 = 20.0
    _keys = {"tol", "c0", "wrap_around"}

    def __init__(self, cell, kpts=numpy.zeros((1, 3))):
        FFTDF.__init__(self, cell, kpts)

        self._fswap = lib.H5TmpFile()

        self._isdf = None
        self._isdf_to_save = None

        self._coul_kpt = None
        self._inpv_kpt = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info("mesh = %s (%d PWs)", self.mesh, numpy.prod(self.mesh))
        log.info("len(kpts) = %d", len(self.kpts))
        log.info("tol = %s", self.tol)
        log.info("c0 = %s", self.c0)
        log.info("wrap_around = %s", self.wrap_around)
        if self._isdf_to_save is not None:
            log.info("isdf_to_save = %s", self._isdf_to_save)
        return self

    select_inpx = select_inpx

    def build_inpv_kpt(self, inpx=None, tol=1e-10):
        log = logger.new_logger(self, self.verbose)
        
        cell = self.cell
        wrap_around = self.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, self.kpts, wrap_around)

        mesh = cell.mesh
        if inpx is None:
            m0 = numpy.asarray(mesh)
            c0 = self.c0
            if numpy.prod(m0) > PARENT_GRID_SIZE_MAX:
                f = PARENT_GRID_SIZE_MAX / numpy.prod(m0)
                m0 = numpy.floor(m0 * f ** (1/3) * 0.5)
                m0 = (m0 * 2 + 1).astype(int)
                log.info("Original mesh %s is too large, reduced to %s as parent grid.", mesh, m0)
            g0 = cell.gen_uniform_grids(m0)
            inpx = select_inpx(self, g0=g0, c0=c0, tol=tol, kpts=kpts)
        else:
            log.debug("Using pre-computed interpolating vectors, c0 is not used")

        log.info("Number of interpolating points is %d.", inpx.shape[0])
        inpv_kpt = cell.pbc_eval_gto("GTOval", inpx, kpts=kpts)
        inpv_kpt = numpy.asarray(inpv_kpt, dtype=numpy.complex128)
        return inpv_kpt

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

        blksize = int(max_memory * 1e6 * 0.1) // (nkpt * nip * 16)
        blksize = max(blksize, 1)
        blksize = min(CONTRACT_BLKSIZE_MAX, blksize, ngrid)

        fswap = self._fswap
        if blksize < ngrid:
            blknum = (ngrid + blksize - 1) // blksize
            blksize = ngrid // blknum + 1
        
        eta_kpt = None
        if fswap is None:
            log.debug("\nIn-core version is used for eta_kpt.")
            log.debug("memory required: %6.2e GB", nkpt * nip * 16 * ngrid / 1e9)
            log.debug("max_memory: %6.2e GB", max_memory / 1e3)
            eta_kpt = numpy.zeros((nkpt, ngrid, nip), dtype=numpy.complex128)
        else:
            log.debug("\nOut-core version is used for eta_kpt.")
            log.debug("disk space required: %6.2e GB.", nkpt * nip * 16 * ngrid / 1e9)
            log.debug("blksize = %d, ngrid = %d", blksize, ngrid)
            log.debug("memory needed for each block:   %6.2e GB", nkpt * nip * 16 * blksize / 1e9)
            log.debug("memory needed for each k-point: %6.2e GB", nip * ngrid * 16 / 1e9)
            log.debug("max_memory: %6.2e GB", max_memory / 1e3)
            eta_kpt = fswap.create_dataset("eta_kpt", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
        
        log.debug("\nComputing eta_kpt")
        info = (lambda s: f"eta_kpt[ %{len(s)}d: %{len(s)}d]")(str(ngrid))
        aoR_loop = self.aoR_loop(grids, kpts, 0, blksize=blksize)
        for ao_etc_kpt, g0, g1 in aoR_loop:
            t0 = (process_clock(), perf_counter())
            ao_kpt = numpy.asarray(ao_etc_kpt[0], dtype=numpy.complex128)

            # eta_kpt_g0g1: (nkpt, nip, g1 - g0)
            eta_kpt_g0g1 = contract(inpv_kpt, ao_kpt, phase)
            eta_kpt_g0g1 = eta_kpt_g0g1.transpose(0, 2, 1)

            eta_kpt[:, g0:g1, :] = eta_kpt_g0g1
            eta_kpt_g0g1 = None

            log.timer(info % (g0, g1), *t0)

        return eta_kpt

    def build_coul_kpt(self, inpv_kpt, eta_kpt):
        log = logger.new_logger(self, self.verbose)
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
        coul_kpt = numpy.zeros((nkpt, nip, nip), dtype=numpy.complex128)

        log.debug("\nComputing coul_kpt")
        info = (lambda s: f"coul_kpt[ %{len(s)}d / {s}]")(str(nkpt))
        for q in range(nkpt):
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

        return coul_kpt
    
    @property
    def inpv_kpt(self):
        if self._inpv_kpt is None:
            if self._isdf is not None:
                self._inpv_kpt = load(self._isdf, "inpv_kpt")
        assert self._inpv_kpt is not None
        return self._inpv_kpt
    
    @property
    def coul_kpt(self):
        if self._coul_kpt is None:
            if self._isdf is not None:
                self._coul_kpt = load(self._isdf, "coul_kpt")
        assert self._coul_kpt is not None
        return self._coul_kpt

    def build(self, inpx=None):
        log = logger.new_logger(self, self.verbose)

        # If a pre-computed ISDF is available, load it
        if self._isdf is not None:
            log.info("Loading ISDF results from %s, skipping build", self._isdf)
            inpv_kpt = load(self._isdf, "inpv_kpt")
            coul_kpt = load(self._isdf, "coul_kpt")
            self._inpv_kpt = inpv_kpt
            self._coul_kpt = coul_kpt
            return inpv_kpt, coul_kpt

        self.dump_flags()
        self.check_sanity()

        # [Step 1]: compute the interpolating functions
        # inpv_kpt is a (nkpt, nip, nao) array
        tol2 = self.tol ** 2
        t0 = (process_clock(), perf_counter())
        inpv_kpt = self.build_inpv_kpt(tol=tol2, inpx=inpx)
        log.timer("building inpv_kpt", *t0)
        nkpt, nip, nao = inpv_kpt.shape

        # [Step 2]: compute the right-hand side of the least-square fitting
        # eta_kpt is a (ngrid, nip, nkpt) array
        t0 = (process_clock(), perf_counter())
        eta_kpt = self.build_eta_kpt(inpv_kpt)
        log.timer("building eta_kpt", *t0)

        # [Step 3]: compute the Coulomb kernel,
        # coul_kpt is a (nkpt, nip, nip) array
        t0 = (process_clock(), perf_counter())
        coul_kpt = self.build_coul_kpt(inpv_kpt, eta_kpt)
        log.timer("building coul_kpt", *t0)

        # [Step 4]: save the results
        self._inpv_kpt = inpv_kpt
        self._coul_kpt = coul_kpt

        if self._isdf_to_save is not None:
            isdf_to_save = self._isdf_to_save
            self._isdf = isdf_to_save
            self.save(isdf_to_save)

        if self._fswap is not None:
            fswap = self._fswap.filename
            self._fswap.close()
            assert not os.path.exists(fswap)
    
    def save(self, isdf_to_save=None):
        log = logger.new_logger(self, self.verbose)

        inpv_kpt = self._inpv_kpt
        coul_kpt = self._coul_kpt
        assert inpv_kpt is not None
        assert coul_kpt is not None

        dump(isdf_to_save, "inpv_kpt", inpv_kpt)
        dump(isdf_to_save, "coul_kpt", coul_kpt)

        nbytes = inpv_kpt.nbytes + coul_kpt.nbytes
        log.info("ISDF results are saved to %s, size = %d MB", isdf_to_save, nbytes / 1e6)

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
    
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        
        assert omega is None and exxdiv is None
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
    a = 1.7834
    lv = a * (numpy.ones((3, 3)) - numpy.eye(3))

    atom  = [['C', (0.0000,     0.0000,   0.0000)]]
    atom += [['C', (a / 2, a / 2, a / 2)]]

    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = atom
    cell.a = lv
    cell.ke_cutoff = 40.0
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.build(dump_input=False)

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
    cc = [10.0, 20.0, 30.0, 40.0, None]
    for c0 in cc:
        from fft import FFTISDF
        scf_obj.with_df = FFTISDF(cell, kpts=kpts)
        scf_obj.with_df.verbose = 10
        scf_obj.with_df.tol = 1e-12

        df_obj = scf_obj.with_df
        df_obj.c0 = c0
        df_obj.verbose = 10
        df_obj.build()

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
