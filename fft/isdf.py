import h5py, functools
from functools import reduce

import numpy, scipy
from scipy.linalg import svd

import pyscf
from pyscf import __config__

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

# fft.isdf_jk
from fft.isdf_jk import get_k_kpts, kpts_to_kmesh, spc_to_kpt, kpt_to_spc
PARENT_GRID_MAXSIZE = getattr(__config__, "isdf_parent_grid_maxsize", 20000)

# Naming convention:
# *_kpt: k-space array, which shapes as (nkpt, x, x)
# *_spc: super-cell stripe array, which shapes as (nspc, x, x)
# *_full: full array, shapes as (nspc * x, nspc * x)
# *_k1, *_k2: the k-space array at specified k-point

def _eta_kpt_g0g1(ao_kpt, inpv_kpt, phase):
    """Compute the eta array in k-space for a given grid range.
    
    Parameters
    ----------
    ao_kpt: numpy.ndarray
        AO value on grids for given slice, shape = (nkpt, ngrid, nao)
    inpv_kpt: numpy.ndarray
        Interpolating vectors, shape = (nkpt, nip, nao)
    phase: numpy.ndarray
        phase factor
    
    Returns
    -------
    eta_kpt_g0g1: numpy.ndarray
        shape = (nkpt, nip, ngrid)
    """
    nkpt, ngrid = ao_kpt.shape[:2]
    nip = inpv_kpt.shape[1]

    t_kpt = inpv_kpt.conj() @ ao_kpt.transpose(0, 2, 1)
    t_kpt = t_kpt.reshape(nkpt, nip, ngrid)
    t_spc = kpt_to_spc(t_kpt, phase)
    
    eta_spc_g0g1 = t_spc ** 2
    eta_kpt_g0g1 = spc_to_kpt(eta_spc_g0g1, phase).conj()

    return eta_kpt_g0g1

def _coul_q(eta, xi, metx, tol=1e-10):
    """Compute the Coulomb kernel in k-space.

    Parameters
    ----------
    eta: numpy.ndarray
        shape = (nip, ngrid)
    xi: numpy.ndarray
        shape = (nip, ngrid)
    metx: numpy.ndarray
        the metrix tensor, shape = (nip, nip)
    tol: float
        The tolerance for the SVD.
    
    Returns
    -------
    coul: numpy.ndarray
        shape = (nip, nip)
    """
    nip, ngrid = eta.shape
    assert xi.shape == (nip, ngrid)
    assert metx.shape == (nip, nip)

    kern = lib.dot(xi.conj(), eta.T)
    kern = kern.reshape(nip, nip)

    u, s, vh = svd(metx, full_matrices=False)
    s2 = s[:, None] * s[None, :]
    m = abs(s2) > tol ** 2

    coul = reduce(lib.dot, (u.conj().T, kern, u))
    coul[m] /= s2[m]
    coul = reduce(lib.dot, (vh.conj().T, coul, vh))

    return coul

def select_inpx(df_obj, g0=None, c0=None, tol=None):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())
    
    pcell = df_obj.cell
    nao = pcell.nao_nr()
    ng = len(g0)

    x0 = pcell.pbc_eval_gto("GTOval", g0)
    m0 = lib.dot(x0.conj(), x0.T) ** 2

    from pyscf.lib.scipy_helper import pivoted_cholesky
    tol2 = tol ** 2
    chol, perm, rank = pivoted_cholesky(m0, tol=tol2)

    nip = rank
    if c0 is not None:
        nip = int(c0 * nao)
    mask = perm[:nip]

    nip = mask.shape[0]
    s0 = numpy.sum(numpy.diag(chol)[:nip])
    s1 = numpy.sum(numpy.diag(chol)[nip:])
    log.info("Parent grid size = %d, selected grid size = %d", ng, nip)
    log.info("Cholesky rank = %d, truncated values = %6.2e, estimated error = %6.2e", rank, s0, s1)

    inpx = g0[mask]
    t1 = log.timer("interpolating functions", *t0)
    return inpx

class InterpolativeSeparableDensityFitting(FFTDF):
    wrap_around = False

    _fswap = None
    _isdf = None
    _isdf_to_save = None

    _coul_kpt = None
    _inpv_kpt = None

    tol = 1e-10
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
        return self
    
    def build(self):
        log = logger.new_logger(self, self.verbose)
        t0 = (process_clock(), perf_counter())
        max_memory = max(2000, self.max_memory - current_memory()[0])

        if self._isdf is not None:
            log.info("Loading ISDF results from %s, skipping build", self._isdf)
            inpv_kpt = load(self._isdf, "inpv_kpt")
            coul_kpt = load(self._isdf, "coul_kpt")
            self._inpv_kpt = inpv_kpt
            self._coul_kpt = coul_kpt
            return inpv_kpt, coul_kpt

        self.dump_flags()
        self.check_sanity()

        cell = self.cell
        nao = cell.nao_nr()
        kpts, kmesh = kpts_to_kmesh(self, self.kpts)
        phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=self.wrap_around)[1]
        nkpt = phase.shape[0]

        tol = self.tol
        inpx = self.inpx
        mesh = cell.mesh
        if inpx is None:
            c0 = self.c0
            m0 = numpy.asarray(mesh)
            if numpy.prod(m0) > PARENT_GRID_MAXSIZE:
                f = PARENT_GRID_MAXSIZE / numpy.prod(m0)
                m0 = numpy.floor(m0 * f ** (1/3) * 0.5).astype(int)
                m0 = m0 * 2 + 1
                log.info("Original mesh %s is too large, reduced to %s", mesh, m0)
            g0 = cell.gen_uniform_grids(m0)
            inpx = select_inpx(self, g0=g0, c0=c0, tol=tol)
        else:
            log.debug("Using pre-computed interpolating vectors, c0 is not used")

        nip = inpx.shape[0]
        assert inpx.shape == (nip, 3)

        grids = self.grids
        ngrid = grids.coords.shape[0]

        if self.blksize is None:
            blksize = max_memory * 1e6 * 0.2 / (nkpt * nip * 16)
            self.blksize = max(1, int(blksize))
        self.blksize = min(self.blksize, ngrid)

        if self.blksize >= ngrid:
            self._fswap = None

        inpv_kpt = cell.pbc_eval_gto("GTOval", inpx, kpts=kpts)
        inpv_kpt = numpy.asarray(inpv_kpt, dtype=numpy.complex128)
        assert inpv_kpt.shape == (nkpt, nip, nao)
        log.debug("nip = %d, nao = %d, cisdf = %6.2f", nip, nao, nip / nao)
        
        # compute metrix tensor in k-space
        t_kpt = inpv_kpt.conj() @ inpv_kpt.transpose(0, 2, 1)
        t_kpt = t_kpt.reshape(nkpt, nip, nip)
        t_spc = kpt_to_spc(t_kpt, phase)
        metx_kpt = spc_to_kpt(t_spc * t_spc, phase)

        # compute Coulomb kernel in k-space
        eta_kpt = None
        fswap = None if self._fswap is None else h5py.File(self._fswap, "w")
        if fswap is not None:
            log.debug("\nOut-core version is used for eta_kpt, disk space required = %6.2e GB.", nkpt * nip * 16 * ngrid / 1e9)
            log.debug("memory used for each block = %6.2e GB, each k-point = %6.2e GB", nkpt * nip * 16 * self.blksize / 1e9, nip * ngrid * 16 / 1e9)
            log.debug("max_memory = %6.2e GB", max_memory / 1e3)
            eta_kpt = fswap.create_dataset("eta_kpt", shape=(nkpt, nip, ngrid), dtype=numpy.complex128)
        else:
            log.debug("\nIn-core version is used for eta_kpt, memory required = %6.2e GB, max_memory = %6.2e GB", nkpt * nip * 16 * ngrid / 1e9, max_memory / 1e3)
            eta_kpt = numpy.zeros((nkpt, nip, ngrid), dtype=numpy.complex128)
        assert eta_kpt is not None
        
        log.debug("\nComputing eta_kpt")
        info = (lambda s: f"eta_kpt[ %{len(s)}d: %{len(s)}d]")(str(ngrid))
        aoR_loop = self.aoR_loop(grids, kpts, deriv=0, blksize=self.blksize)
        for ig, (ao_etc_kpt, g0, g1) in enumerate(aoR_loop):
            t0 = (process_clock(), perf_counter())
            ao_kpt = numpy.asarray(ao_etc_kpt[0])
            eta_kpt_g0g1 = _eta_kpt_g0g1(ao_kpt, inpv_kpt, phase)
            eta_kpt[:, :, g0:g1] = eta_kpt_g0g1
            t1 = log.timer(info % (g0, g1), *t0)

        log.debug("\nComputing coul_kpt")
        info = (lambda s: f"coul_kpt[ %{len(s)}d / {s}]")(str(nkpt))
        v0 = cell.get_Gv(mesh)
        coul_kpt = numpy.zeros((nkpt, nip, nip), dtype=numpy.complex128)
        for q in range(nkpt):
            t0 = (process_clock(), perf_counter())

            eta_q = eta_kpt[q]
            metx_q = metx_kpt[q]

            tq = numpy.dot(grids.coords, kpts[q])
            fq = numpy.exp(-1j * tq) 
            vq  = fft(eta_q * fq, mesh)
            vq *= pbctools.get_coulG(cell, k=kpts[q], Gv=v0, mesh=mesh)
            vq *= cell.vol / ngrid
            xi_q = ifft(vq, mesh) * fq.conj()

            coul_q = _coul_q(eta_q, xi_q, metx_q, tol=tol)
            coul_kpt[q] = coul_q

            t1 = log.timer(info % (q + 1), *t0)

        self._coul_kpt = coul_kpt
        self._inpv_kpt = inpv_kpt

        if self._isdf_to_save is not None:
            self._isdf = self._isdf_to_save

        if self._isdf is not None:
            dump(self._isdf, "coul_kpt", coul_kpt)
            dump(self._isdf, "inpv_kpt", inpv_kpt)

        t1 = log.timer("building ISDF", *t0)
        if fswap is not None:
            fswap.close()

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

ISDF = FFTISDF = InterpolativeSeparableDensityFitting

if __name__ == "__main__":
    cell = pyscf.pbc.gto.M(
    a = numpy.ones((3, 3)) * 3.5668 - numpy.eye(3) * 3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917''',
    basis = 'gth-dzvp', pseudo = 'gth-pade',
    verbose = 4, ke_cutoff = 40.0
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
        # inpx = df_obj.get_inpx(g0=g0, c0=c0, tol=1e-10)
        df_obj.c0 = c0
        df_obj.build(verbose=10)

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

