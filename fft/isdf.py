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
from pyscf.lib.scipy_helper import pivoted_cholesky
from pyscf.pbc.lib.kpts_helper import get_kconserv

# pyscf.pbc
from pyscf.pbc import tools as pbctools
from pyscf.pbc.df.aft import _check_kpts
from pyscf.pbc.df.fft import FFTDF

from fft import isdf_ao2mo
from fft.isdf_jk import spc_to_kpt, kpts_to_kmesh
from fft.isdf_jk import kpt_to_spc, get_phase_factor

from pyscf import __config__
from pyscf.pbc.dft.gen_grid import BLKSIZE
CHOLESKY_TOL = getattr(__config__, "fftisdf_cholesky_tol", 1e-20)
CHOLESKY_MAX_SIZE = getattr(__config__, "fftisdf_cholesky_max_size", 20000)
CONTRACT_MAX_SIZE = getattr(__config__, "fftisdf_contract_max_size", 20000)

# Naming convention:
# *_kpt: k-space array, which shapes as (nkpt, x, x)
# *_spc: super-cell stripe array, which shapes as (nspc, x, x)

def contract(f_kpt, g_kpt, phase):
    r"""Contract two k-space arrays with the following steps:
        [1] Matrix multiplication:       T_{ml}^k = \sum_n F_{mn}^k* G_{ln}^k
        [2] kspace->supercell transform. T_{ml}^s <- T_{ml}^k
        [3] Element-wise square:         X_{ml}^s = T_{ml}^s T_{ml}^s
        [4] supercell->kspace transform. X_{ml}^k <- X_{ml}^s
    
    Args:
        f_kpt (ndarray): First k-space array, shape (k, m, n)
        g_kpt (ndarray): Second k-space array, shape (k, l, n)  
        phase (ndarray): Phase factors for k-space transforms
        
    Returns:
        ndarray: Contracted result in k-space, shape (k, m, l)
    """
    k, m, n = f_kpt.shape
    l = g_kpt.shape[1]
    assert f_kpt.shape == (k, m, n)
    assert g_kpt.shape == (k, l, n)

    # [1] Matrix multiplication
    t_kpt = [lib.dot(fk.conj(), gk.T) for fk, gk in zip(f_kpt, g_kpt)]
    t_kpt = numpy.array(t_kpt).reshape(k, m, l)

    # [2] kspace->supercell transform
    t_spc = kpt_to_spc(t_kpt, phase)

    # [3] Element-wise square
    x_spc = t_spc * t_spc

    # [4] supercell->kspace transform
    x_kpt = spc_to_kpt(x_spc, phase)
    return x_kpt

def lstsq(a, b, tol=1e-10):
    r"""
    Solve the Hermitian sandwich least-squares problem using SVD.
        A dot X dot A ~ B, where A is Hermitian

    Args:
        a (ndarray): Hermitian matrix, shape (m, m)
        b (ndarray): Right-hand side, shape (m, n)
        tol (float): Tolerance for SVD

    Returns:
    """
    # [1] SVD of A
    u, s, vh = svd(a, full_matrices=False)

    # [2] Compute R[i, j] = 1 / S[i] * S[j] 
    # if S[i] * S[j] > tol, otherwise 0
    r = s[None, :] * s[:, None]
    m = abs(r) > tol * tol

    # [3] Compute T = (Uh dot B dot U) * R
    bu = lib.dot(b, u)
    uh = u.conj().T
    t = lib.dot(uh, bu)
    t[m] /= r[m]
    
    # [4] Compute X = V dot T dot Vh
    v = vh.conj().T
    vt = lib.dot(v, t)
    x = lib.dot(vt, vh)

    return x, numpy.sum(s > tol)

def compute_blksize(ntot, nmax=2000, chunk=BLKSIZE):
    """Participate the grid into slices. Output an integer
    which corresponds to the hlksize for the loop with the
    following condition:
        - each slice is smaller than bmax
        - blksize is a multiple of blks
        - try to make each slice equal size
    """
    blksize_max = nmax // chunk * chunk
    blknum_min = (ntot + blksize_max - 1) // blksize_max
    blksize = (ntot + blknum_min - 1) // blknum_min
    blksize = (blksize + chunk - 1) // chunk * chunk
    blksize = min(blksize, blksize_max)
    return blksize

def compute_metx(f_kpt):
    """This is a helper function to compute the metric
    tensor in the reference cell. Equivalent to the following
    code:
        metx_kpt = contract(f_kpt, f_kpt, phase) # some phase
        metx_spc = kpt_to_spc(metx_kpt, phase)
        metx = metx_spc[0]
    """
    nk, nx = f_kpt.shape[:2]
    f_kpt = f_kpt.transpose(1, 0, 2).reshape(nx, -1)
    metx = lib.dot(f_kpt.conj(), f_kpt.T)
    metx = metx * metx.conj()
    return metx.real / nk

def select_interpolating_points(df_obj, cisdf=None):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())
    
    cell = df_obj.cell
    nao = cell.nao_nr()
    grids = df_obj.grids
    ngrid = grids.coords.shape[0]

    kpts = df_obj.kpts
    phase = get_phase_factor(cell, kpts)
    nimg, nkpt = phase.shape

    ix = numpy.arange(ngrid)
    if ngrid > CHOLESKY_MAX_SIZE:
        max_memory = max(2000, df_obj.max_memory - current_memory()[0])
        blksize_max = int(max_memory * 1e6 * 0.5) // (nkpt * nao * 16)
        blksize_max = min(blksize_max, CHOLESKY_MAX_SIZE)
        blksize = compute_blksize(ngrid, blksize_max, BLKSIZE)

        log.debug("Select interpolating points with local Cholesky decomposition:")
        log.debug("CHOLESKY_MAX_SIZE = %d, BLKSIZE = %d", CHOLESKY_MAX_SIZE, BLKSIZE)
        log.debug("ngrid = %d, blksize = %d", ngrid, blksize)
        log.debug("ngrid // blksize = %d, ngrid %% blksize = %d", ngrid // blksize, ngrid % blksize)

        info = (lambda s: f"Cholesky decomposition for grids [%{len(s)+1}d, %{len(s)+1}d]")(str(ngrid))
        block_loop = df_obj.gen_block_loop(blksize=blksize)
    
        weight = numpy.zeros(ngrid)        
        for ao_etc_kpt, g0, g1 in block_loop:
            ao_g0g1_kpt = numpy.asarray(ao_etc_kpt[0], dtype=numpy.complex128)

            metx_g0g1 = compute_metx(ao_g0g1_kpt)
            chol, perm, rank = pivoted_cholesky(metx_g0g1, tol=CHOLESKY_TOL)

            weight[g0+perm] = numpy.diag(chol)
            log.debug(info % (g0, g1) + ", rank = %d" % rank)

        ix = numpy.argsort(weight)[-CHOLESKY_MAX_SIZE:]
        ix = ix[weight[ix] > CHOLESKY_TOL]
        ix = numpy.sort(ix)
    
    coord = grids.coords[ix]
    phi_kpt = cell.pbc_eval_gto("GTOval", coord, kpts=kpts)
    phi_kpt = numpy.asarray(phi_kpt, dtype=numpy.complex128)
    metx = compute_metx(phi_kpt)

    chol, perm, rank = pivoted_cholesky(metx, tol=CHOLESKY_TOL)
    log.info("Parent grid size = %d, Cholesky rank = %d", ngrid, rank)

    nip = rank
    if cisdf is not None:
        nip = min(nip, int(cisdf * nao))
    else:
        log.info("Using all Cholesky vectors as interpolating points.")

    log.info("nao = %d, nip = %d, cisdf = %6.2f", nao, nip, nip / nao)
    log.info("Largest Cholesky weight:   %6.2e", numpy.diag(chol)[0])
    log.info("Smallest remaining weight: %6.2e", numpy.diag(chol)[nip - 1])
    log.info("Largest discarded weight:  %6.2e", numpy.diag(chol)[nip])
    log.info("Total remaining weight:    %6.2e", numpy.diag(chol)[:nip].sum())
    log.info("Total discarded weight:    %6.2e", numpy.diag(chol)[nip:].sum())

    log.timer("selecting interpolating points", *t0)
    mask = perm[:nip]
    return coord[mask]

class InterpolativeSeparableDensityFitting(FFTDF):
    tol = 1e-8
    _keys = {"tol", "kconserv2", "kconserv3"}

    def __init__(self, cell, kpts=numpy.zeros((1, 3))):
        FFTDF.__init__(self, cell, kpts)

        kconserv = get_kconserv(cell, kpts)
        self.kconserv3 = kconserv
        self.kconserv2 = kconserv[:, :, 0].T

        self._fswap = lib.H5TmpFile()

        self._isdf = None
        self._isdf_to_save = None

        self._coul_kpt = None
        self._inpv_kpt = None

    get_eri = isdf_ao2mo.get_ao_eri
    get_ao_eri = isdf_ao2mo.get_ao_eri
    get_mo_eri = isdf_ao2mo.get_mo_eri
    ao2mo = isdf_ao2mo.ao2mo_7d
    ao2mo_spc = isdf_ao2mo.ao2mo_spc
    ao2mo_7d = isdf_ao2mo.ao2mo_7d

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        
        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info("mesh = %s (%d PWs)", self.mesh, numpy.prod(self.mesh))
        log.info("lstsq tol = %s", self.tol)

        kpts, kmesh = kpts_to_kmesh(self.cell, self.kpts)
        log.info("kmesh = %s", kmesh)

        inpv_kpt = self.inpv_kpt
        nip, nao = inpv_kpt.shape[1:]
        log.info("nip = %d", nip)
        log.info("nao = %d", nao)
        log.info("cisdf = %6.2f", nip / nao)

        if self._isdf_to_save is not None:
            log.info("isdf_to_save = %s", self._isdf_to_save)
        return self

    def build_inpv_kpt(self, cisdf=10.0):
        cell = self.cell
        kpts = self.kpts
        inpx = select_interpolating_points(self, cisdf=cisdf)
        inpv_kpt = cell.pbc_eval_gto("GTOval", inpx, kpts=kpts)
        inpv_kpt = numpy.asarray(inpv_kpt, dtype=numpy.complex128)
        return inpv_kpt

    def build_eta_kpt(self, inpv_kpt):
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
        
        eta_kpt = None
        fswap = self._fswap

        shape = (nkpt, ngrid, nip)
        dtype = numpy.complex128

        if fswap is None:
            log.debug("\nIn-core version is used for eta_kpt.")
            log.debug("approximate memory required: %6.2e GB", numpy.prod(shape) * 16 / 1e9)
            log.debug("max_memory: %6.2e GB", max_memory / 1e3)
            eta_kpt = numpy.zeros(shape, dtype=dtype)
        else:
            log.debug("\nOut-core version is used for eta_kpt.")
            log.debug("disk space required: %6.2e GB", numpy.prod(shape) * 16 / 1e9)
            log.debug("blksize = %d, ngrid = %d", blksize, ngrid)
            log.debug("approximate memory needed for each block:   %6.2e GB", nkpt * nip * 16 * blksize / 1e9)
            log.debug("approximate memory needed for each k-point: %6.2e GB", nip * ngrid * 16 / 1e9)
            log.debug("max_memory: %6.2e GB", max_memory / 1e3)
            eta_kpt = fswap.create_dataset("eta_kpt", shape=shape, dtype=dtype)
        
        log.debug("\nComputing eta_kpt")
        info = (lambda s: f"eta_kpt[ %{len(s)}d: %{len(s)}d]")(str(ngrid))
        block_loop = self.gen_block_loop(blksize=blksize)
        for ao_etc_kpt, g0, g1 in block_loop:
            t0 = (process_clock(), perf_counter())
            ao_kpt = numpy.asarray(ao_etc_kpt[0], dtype=numpy.complex128)

            # eta_kpt_g0g1: (nkpt, nip, g1 - g0)
            eta_kpt_g0g1 = contract(ao_kpt, inpv_kpt, phase)
            eta_kpt_g0g1 = eta_kpt_g0g1.conj()
            assert eta_kpt_g0g1.shape == (nkpt, g1 - g0, nip)

            eta_kpt[:, g0:g1, :] = eta_kpt_g0g1
            eta_kpt_g0g1 = None

            log.timer(info % (g0, g1), *t0)

        return eta_kpt

    def build_coul_kpt(self, inpv_kpt, eta_kpt):
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
        coul_kpt = numpy.zeros((nkpt, nip, nip), dtype=numpy.complex128)

        log.debug("\nComputing coul_kpt")
        info = (lambda s: f"coul_kpt[ %{len(s)}d / {s}]")(str(nkpt))
        for q in range(nkpt):
            t0 = (process_clock(), perf_counter())
            
            fq = numpy.exp(-1j * coord @ kpts[q])
            vq = pbctools.get_coulG(cell, k=kpts[q], exx=False, Gv=v0, mesh=mesh)
            vq *= cell.vol / ngrid
            lq = eta_kpt[q].T * fq
            wq = pbctools.fft(lq, mesh)
            rq = pbctools.ifft(wq * vq, mesh)
            kern_q = lib.dot(lq, rq.conj().T) / numpy.sqrt(ngrid)
            lq = rq = wq = None

            metx_q = metx_kpt[q]
            res = lstsq(metx_q, kern_q, tol=tol)
            coul_q = res[0]
            if log.verbose >= logger.DEBUG1:
                err = metx_q @ coul_q @ metx_q - kern_q
                err = abs(err).max() / abs(kern_q).max()
                log.debug("\nMetric tensor rank: %d / %d, lstsq error: %6.2e", res[1], nip, err)

            coul_q *= numpy.sqrt(ngrid)
            coul_q = (coul_q + coul_q.conj().T) / 2

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

    def build(self, cisdf=10.0):
        log = logger.new_logger(self, self.verbose)

        # If a pre-computed ISDF is available, load it
        if self._isdf is not None:
            isdf_to_read = self._isdf
            assert os.path.exists(isdf_to_read)

            log.info("Loading ISDF results from %s, skipping build", isdf_to_read)
            inpv_kpt = load(isdf_to_read, "inpv_kpt")
            coul_kpt = load(isdf_to_read, "coul_kpt")
            self._inpv_kpt = inpv_kpt
            self._coul_kpt = coul_kpt
            return inpv_kpt, coul_kpt
        
        self.check_sanity()

        # [Step 1]: compute the interpolating functions
        # inpv_kpt is a (nkpt, nip, nao) array
        inpv_kpt = self._inpv_kpt
        if inpv_kpt is not None:
            log.debug("Using pre-computed interpolating vectors, c0 is not used")
        else:
            inpv_kpt = self.build_inpv_kpt(cisdf=cisdf)
            self._inpv_kpt = inpv_kpt

        self.dump_flags()
        
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
        self._finalize()
    
    def _finalize(self):
        log = logger.new_logger(self, self.verbose)

        if self._fswap is not None:
            fswap = self._fswap.filename
            self._fswap.close()
            self._fswap = None

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

    def gen_block_loop(self, deriv=0, blksize=None):
        grids = self.grids
        assert grids.non0tab is None

        cell = grids.cell
        assert cell.dimension == 3
        
        kpts = self.kpts
        if not isinstance(kpts, numpy.ndarray):
            kpts = kpts.kpts
        kpts = numpy.asarray(kpts)

        ni = self._numint
        nao = cell.nao_nr()
        max_memory = self.max_memory - current_memory()[0]

        if blksize is not None:
            assert blksize % BLKSIZE == 0

        block_loop = ni.block_loop(
            cell, grids, nao, deriv, 
            kpts, blksize=blksize, 
            max_memory=max_memory,
            )
        
        g0 = g1 = 0
        for ao_etc_kpt in block_loop:
            g0 = g1
            g1 += ao_etc_kpt[4].shape[0]
            yield ao_etc_kpt, g0, g1
    
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):

        assert omega is None
        kpts, is_single_kpt = _check_kpts(self, kpts)

        vj = vk = None
        if with_k:
            from fft.isdf_jk import get_k_kpts
            vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)

        if with_j:
            from pyscf.pbc.df.fft_jk import get_j_kpts
            vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)

        return vj, vk

ISDF = FFTISDF = InterpolativeSeparableDensityFitting

if __name__ == "__main__":
    a = 1.7834 # unit Angstrom
    lv = a * (numpy.ones((3, 3)) - numpy.eye(3))

    atom  = [['C', (0.0000, 0.0000, 0.0000)]]
    atom += [['C', ( a / 2,  a / 2,  a / 2)]]

    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = atom
    cell.a = lv
    cell.ke_cutoff = 40.0
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.unit = "A"
    cell.verbose = 0
    cell.build(dump_input=False)

    nao = cell.nao_nr()
    # kmesh = [2, 2, 2] # 4, 4, 4]
    kmesh = [4, 4, 4]
    nkpt = nspc = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.conv_tol = 1e-8
    scf_obj.verbose = 0
    dm_kpts = None

    log = logger.new_logger(None, 5)

    res = []
    for c0 in [5.0, 10.0, 15.0, 20.0]:
        from fft import FFTISDF
        scf_obj.with_df = FFTISDF(cell, kpts=kpts)
        scf_obj.with_df.verbose = 10

        df_obj = scf_obj.with_df
        df_obj.build(cisdf=c0)

        scf_obj.kernel(dm_kpts)
        dm_kpts = scf_obj.make_rdm1()

        e_tot = scf_obj.energy_tot(dm_kpts)
        res.append((c0, e_tot))

    from pyscf.pbc.df.fft import FFTDF
    scf_obj.with_df = FFTDF(cell, kpts)
    scf_obj.with_df.verbose = 0
    scf_obj.kernel(dm_kpts)
    e_ref = scf_obj.e_tot

    for ires, (c0, e_sol) in enumerate(res):
        err = abs(e_sol - e_ref) / abs(e_ref)
        print("-> FFTISDF c0 = %6s, ene_err = % 6.2e" % (c0, err))