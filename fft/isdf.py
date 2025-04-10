import os, sys, h5py
from itertools import product
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
from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools.pbc import fft, ifft
from pyscf.pbc.tools.k2gamma import get_phase
from pyscf.pbc.lib import kpts_helper

# fft.isdf_jk
from fft.isdf_jk import get_k_kpts, kpts_to_kmesh, spc_to_kpt, kpt_to_spc

PARENT_GRID_MAXSIZE = getattr(__config__, "isdf_parent_grid_maxsize", 10000)

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

    t_kpt = f_kpt.conj() @ g_kpt.transpose(0, 2, 1)
    t_kpt = t_kpt.reshape(nk, m, l)
    t_spc = kpt_to_spc(t_kpt, phase)

    x_spc = t_spc * t_spc
    x_kpt = spc_to_kpt(x_spc, phase)
    return x_kpt

def member(kpt, kpts):
    ind = kpts_helper.member(kpt, kpts)
    assert len(ind) == 1, f"search for {kpt} in {kpts} returns {ind}"
    return ind[0]

def lstsq(a, b, tol=1e-10):
    u, s, vh = svd(a, full_matrices=False)
    uh = u.conj().T
    v = vh.conj().T
    s2 = s[None, :] * s[:, None]
    
    mask = abs(s2) > tol
    x = reduce(numpy.dot, (uh, b, u))
    x[mask] /= s2[mask]
    x = reduce(numpy.dot, (v, x, vh))

    return x

def select_inpx(df_obj, g0=None, c0=None, kpts=None, tol=1e-10):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())
    
    cell = df_obj.cell
    nao = cell.nao_nr()
    ng = g0.shape[0]

    if kpts is None:
        kpts = numpy.zeros((1, 3))

    m0 = numpy.zeros((ng, ng), dtype=numpy.complex128)
    for kpt in kpts:
        x0 = cell.pbc_eval_gto("GTOval", g0, kpts=kpt)
        m0 += lib.dot(x0.conj(), x0.T) ** 2
    m0 = m0.real

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
    t1 = log.timer("interpolating points", *t0)
    return inpx

def build(df_obj, tol=1e-10):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())
    max_memory = max(2000, df_obj.max_memory - current_memory()[0])

    df_obj.dump_flags()
    df_obj.check_sanity()

    cell = df_obj.cell
    nao = cell.nao_nr()

    mesh = cell.mesh
    inpx = df_obj.inpx
    if inpx is None:
        m0 = numpy.asarray(mesh)
        c0 = df_obj.c0
        if numpy.prod(m0) > PARENT_GRID_MAXSIZE:
            f = PARENT_GRID_MAXSIZE / numpy.prod(m0)
            m0 = numpy.floor(m0 * f ** (1/3) * 0.5)
            m0 = (m0 * 2 + 1).astype(int)
            log.info("Original mesh %s is too large, reduced to %s as parent grid.", mesh, m0)
        g0 = cell.gen_uniform_grids(m0)
        inpx = select_inpx(df_obj, g0=g0, c0=c0, tol=tol)
    else:
        log.debug("Using pre-computed interpolating vectors, c0 is not used")
    df_obj.inpx = inpx

    nip = inpx.shape[0]
    assert inpx.shape == (nip, 3)
    df_obj.inpx = inpx

    kpts, kmesh = kpts_to_kmesh(df_obj, df_obj.kpts)
    nkpt = kpts.shape[0]
    ngrid = df_obj.grids.coords.shape[0]

    if df_obj.blksize is None:
        blksize = max_memory * 1e6 * 0.2 / (nkpt * nip * 16)
        df_obj.blksize = max(1, int(blksize))
    df_obj.blksize = min(df_obj.blksize, ngrid)

    if df_obj.blksize >= ngrid:
        df_obj._fswap = None

    inpv_kpt = cell.pbc_eval_gto("GTOval", inpx, kpts=kpts)
    inpv_kpt = numpy.asarray(inpv_kpt, dtype=numpy.complex128)
    df_obj._inpv_kpt = inpv_kpt
    
    return df_obj
    
class InterpolativeSeparableDensityFitting(FFTDF):
    wrap_around = False

    _fswap = None
    _isdf = None
    _isdf_to_save = None

    _coul_kpt = None
    _inpv_kpt = None

    tol = 1e-8
    blksize = None
    c0 = None
    inpx = None

    _keys = {"blksize", "tol", "c0", "inpx"}

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
        log.info("tol = %s", self.tol)
        return self

    select_inpx = select_inpx
    
    def build(self):
        log = logger.new_logger(self, self.verbose)
        max_memory = max(2000, self.max_memory - current_memory()[0])

        # If a pre-computed ISDF is available, load it
        if self._isdf is not None:
            log.info("Loading ISDF results from %s, skipping build", self._isdf)
            inpv_kpt = load(self._isdf, "inpv_kpt")
            coul_kpt = load(self._isdf, "coul_kpt")
            self._inpv_kpt = inpv_kpt
            self._coul_kpt = coul_kpt
            return inpv_kpt, coul_kpt

        tol2 = self.tol ** 2
        build(self, tol=tol2)

        grids = self.grids
        ngrid = grids.coords.shape[0]

        # [Step 1]: read the pre-computed interpolating functions
        # inpv_kpt is a (nkpt, nip, nao) array
        inpv_kpt = self._inpv_kpt
        nkpt, nip, nao = inpv_kpt.shape

        # Setup the scratch file for eta_kpt if needed
        fswap = None if self._fswap is None else h5py.File(self._fswap, "w")
        if fswap is None:
            log.debug("\nIn-core version is used for eta_kpt.")
            log.debug("memory required: %6.2e GB", nkpt * nip * 16 * ngrid / 1e9)
            log.debug("max_memory: %6.2e GB", max_memory / 1e3)
        else:
            log.debug("\nOut-core version is used for eta_kpt.")
            log.debug("disk space required: %6.2e GB.", nkpt * nip * 16 * ngrid / 1e9)
            log.debug("memory needed for each block:   %6.2e GB", nkpt * nip * 16 * self.blksize / 1e9)
            log.debug("memory needed for each k-point: %6.2e GB", nip * ngrid * 16 / 1e9)
            log.debug("max_memory: %6.2e GB", max_memory / 1e3)

        cell = self.cell
        mesh = cell.mesh
        v0 = cell.get_Gv(mesh)

        kpts, kmesh = kpts_to_kmesh(self, self.kpts)
        phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=self.wrap_around)[1]

        # [Step 2]: compute the metric tensor,
        # metx_kpt is a (nkpt, nip, nip) array
        metx_kpt = contract(inpv_kpt, inpv_kpt, phase)
        assert metx_kpt.shape == (nkpt, nip, nip)

        # [Step 3]: compute the right-hand side of the least-square fitting
        # eta_kpt is a (nkpt, nip, ngrid) array
        eta_kpt = None
        if fswap is not None:
            eta_kpt = fswap.create_dataset("eta_kpt", shape=(nkpt, nip, ngrid), dtype=numpy.complex128)
        else:
            eta_kpt = numpy.zeros((nkpt, nip, ngrid), dtype=numpy.complex128)
        
        log.debug("\nComputing eta_kpt")
        info = (lambda s: f"eta_kpt[ %{len(s)}d: %{len(s)}d]")(str(ngrid))
        aoR_loop = self.aoR_loop(grids, kpts, 0, blksize=self.blksize)
        for ig, (ao_etc_kpt, g0, g1) in enumerate(aoR_loop):
            t0 = (process_clock(), perf_counter())
            ao_kpt = numpy.asarray(ao_etc_kpt[0])
            eta_kpt_g0g1 = contract(inpv_kpt, ao_kpt, phase)
            eta_kpt[:, :, g0:g1] = eta_kpt_g0g1
            t1 = log.timer(info % (g0, g1), *t0)

        # [Step 4]: compute the Coulomb kernel,
        # coul_kpt is a (nkpt, nip, nip) array
        coul_kpt = numpy.zeros((nkpt, nip, nip), dtype=numpy.complex128)

        log.debug("\nComputing coul_kpt")
        info = (lambda s: f"coul_kpt[ %{len(s)}d / {s}]")(str(nkpt))
        for q in range(nkpt):
            t0 = (process_clock(), perf_counter())
            
            tq = numpy.dot(grids.coords, kpts[q])
            fq = numpy.exp(-1j * tq) 
            vq = pbctools.get_coulG(cell, k=kpts[q], Gv=v0, mesh=mesh)
            vq *= cell.vol / ngrid

            lhs = eta_kpt[q] * fq
            wq  = fft(lhs, mesh)
            rhs = ifft(wq * vq, mesh)
            kern_q = lib.dot(lhs, rhs.conj().T)

            metx_q = metx_kpt[q]
            coul_q = lstsq(metx_q, kern_q, tol=tol2)
            coul_kpt[q] = coul_q

            t1 = log.timer(info % (q + 1), *t0)

        self._inpv_kpt = inpv_kpt
        self._coul_kpt = coul_kpt

        if self._isdf_to_save is not None:
            self._isdf = self._isdf_to_save

        if fswap is not None:
            fswap.close()

        if self._isdf is None and self._fswap is not None:
            self._isdf = self._fswap

        if self._isdf is not None:
            dump(self._isdf, "coul_kpt", coul_kpt)
            dump(self._isdf, "inpv_kpt", inpv_kpt)
            log.info("ISDF results are saved to %s", self._isdf)

        return inpv_kpt, coul_kpt

    
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
        from pyscf.pbc.df.aft import _check_kpts
        kpts, is_single_kpt = _check_kpts(self, kpts)

        vj = vk = None
        if with_k:
            vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            from pyscf.pbc.df.fft_jk import get_j_kpts
            vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)

        return vj, vk
    
    def get_ao_eri(self, kpts, compact=False):
        cell = self.cell
        from pyscf.pbc.lib.kpts_helper import get_kconserv, get_kconserv_ria
        from pyscf.pbc.lib.kpts_helper import loop_kkk
        kconserv2 = get_kconserv_ria(cell, self.kpts)
        kconserv3 = get_kconserv(cell, self.kpts)

        km, kn, kl, ks = [member(kpt, self.kpts) for kpt in kpts]
        is_conserved = (kconserv3[km, kn, kl] == ks)
        if not is_conserved:
            raise ValueError("kpts are not conserved")

        assert compact is False
        
        kq = kconserv2[km, kn]
        jq = self._coul_kpt[kq]

        inpv_kpt = self._inpv_kpt
        nkpt, nip, nao = inpv_kpt.shape
        nao2 = nao * nao
        rho_mn = inpv_kpt[km].conj().reshape(-1, nao, 1) * inpv_kpt[kn].reshape(-1, 1, nao)
        rho_mn = rho_mn.reshape(-1, nao2)

        rho_ls = inpv_kpt[kl].conj().reshape(-1, nao, 1) * inpv_kpt[ks].reshape(-1, 1, nao)
        rho_ls = rho_ls.reshape(-1, nao2)

        vq = lib.dot(rho_mn.T, jq)
        eri_ao = lib.dot(vq, rho_ls)
        return eri_ao

    def ao2mo_7d(self, mo_coeff_kpts, kpts=None):
        if kpts is None:
            kpts = self.kpts

        kpts = numpy.asarray(kpts)
        assert numpy.all(kpts == self.kpts)

        # get the Coulomb kernel
        inpv_kpt = self._inpv_kpt
        coul_kpt = self._coul_kpt
        nkpt, nip, nao = inpv_kpt.shape

        nmo = mo_coeff_kpts.shape[2]
        assert mo_coeff_kpts.shape == (nkpt, nao, nmo)

        nmo2 = nmo * nmo
        shape = (nkpt, ) * 3 + (nmo2, ) * 2
        eri_7d = numpy.zeros(shape, dtype=numpy.complex128)

        cell = self.cell
        from pyscf.pbc.lib.kpts_helper import get_kconserv, get_kconserv_ria
        kconserv2 = get_kconserv_ria(cell, kpts)
        kconserv3 = get_kconserv(cell, kpts)

        inpv_kpt = numpy.einsum("kIm,kmp->kIp", self._inpv_kpt, mo_coeff_kpts)

        for km, kn in product(range(nkpt), repeat=2):
            kq = kconserv2[km, kn]
            jq = coul_kpt[kq]

            rho_mn = inpv_kpt[km].conj().reshape(-1, nmo, 1) * inpv_kpt[kn].reshape(-1, 1, nmo)
            rho_mn = rho_mn.reshape(-1, nmo2)
            vq = lib.dot(rho_mn.T, jq)

            for kl in range(nkpt):
                ks = kconserv3[km, kn, kl]
                rho_ls = inpv_kpt[kl].conj().reshape(-1, nmo, 1) * inpv_kpt[ks].reshape(-1, 1, nmo)
                rho_ls = rho_ls.reshape(-1, nmo2)
                eri_7d[km, kn, kl] = lib.dot(vq, rho_ls)
        
        return eri_7d
        

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
