import os, sys
import numpy, scipy

import pyscf
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero

from pyscf.pbc.tools.k2gamma import get_phase, kpts_to_kmesh as pyscf_kpts_to_kmesh
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 2000))

def kpts_to_kmesh(df_obj, kpts):
    kmesh = pyscf_kpts_to_kmesh(df_obj.cell, kpts)
    assert numpy.allclose(kpts, df_obj.cell.get_kpts(kmesh))
    return kpts, kmesh

def spc_to_kpt(m_spc, phase):
    """Convert a matrix from the stripe form (in super-cell)
    to the k-space form.
    """
    nspc, nkpt = phase.shape
    m_kpt = lib.dot(phase.conj().T, m_spc.reshape(nspc, -1))
    return m_kpt.reshape(m_spc.shape)

def kpt_to_spc(m_kpt, phase):
    """Convert a matrix from the k-space form to
    stripe form (in super-cell).
    """
    nspc, nkpt = phase.shape
    m_spc = lib.dot(phase, m_kpt.reshape(nkpt, -1))
    m_spc = m_spc.reshape(m_kpt.shape)
    return m_spc.real

def get_j_kpts(df_obj, dm_kpts, hermi=1, kpts=numpy.zeros((1, 3)), kpts_band=None,
               exxdiv=None):
    """
    Get the J matrix for a set of k-points.
    
    Args:
        df_obj: The FFT-ISDF object. 
        dm_kpts: Density matrices at each k-point.
        hermi: Whether the density matrices are Hermitian.
        kpts: The k-points to calculate J for.
        kpts_band: The k-points of the bands.
        exxdiv: The divergence of the exchange functional (ignored).
        
    Returns:
        The J matrix at the specified k-points.
    """
    cell = df_obj.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1

    pcell = df_obj.cell
    # check if kpts is consistent with df_obj.kpts
    assert numpy.allclose(kpts, df_obj.kpts)
    kpts, kmesh = kpts_to_kmesh(df_obj, df_obj.kpts)

    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, df_obj.kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )

    nao = pcell.nao_nr()
    nkpt = nspc = numpy.prod(kmesh)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpt, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"

    inpv_kpt = df_obj._inpv_kpt
    coul_kpt = df_obj._coul_kpt

    nip = inpv_kpt.shape[1]
    assert inpv_kpt.shape == (nkpt, nip, nao)
    assert coul_kpt.shape == (nkpt, nip, nip)

    coul0 = coul_kpt[0]
    assert coul0.shape == (nip, nip)

    rho = numpy.einsum("kIm,kIn,xkmn->xI", inpv_kpt, inpv_kpt.conj(), dms, optimize=True)
    rho *= 1.0 / nkpt
    assert rho.shape == (nset, nip)

    v0 = numpy.einsum("IJ,xJ->xI", coul0, rho, optimize=True)
    assert v0.shape == (nset, nip)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"

    vj_kpts = numpy.einsum("kIm,kIn,xI->xkmn", inpv_kpt.conj(), inpv_kpt, v0, optimize=True)
    assert vj_kpts.shape == (nset, nkpt, nao, nao)

    if is_zero(kpts_band):
        vj_kpts = vj_kpts.real
    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def get_k_kpts(df_obj, dm_kpts, hermi=1, kpts=numpy.zeros((1, 3)), kpts_band=None,
               exxdiv=None):
    cell = df_obj.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1

    pcell = df_obj.cell
    # check if kpts is consistent with df_obj.kpts
    assert numpy.allclose(kpts, df_obj.kpts)
    kpts, kmesh = kpts_to_kmesh(df_obj, df_obj.kpts)
    
    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, df_obj.kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )

    nao = pcell.nao_nr()
    nkpt = nspc = numpy.prod(kmesh)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpt, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"
    assert exxdiv is None, f"exxdiv = {exxdiv}"

    inpv_kpt = df_obj._inpv_kpt
    coul_kpt = df_obj._coul_kpt

    nip = inpv_kpt.shape[1]
    coul_spc = kpt_to_spc(coul_kpt, phase)
    coul_spc = coul_spc * numpy.sqrt(nkpt)
    coul_spc = coul_spc.reshape(nspc, nip, nip)
    
    assert inpv_kpt.shape == (nkpt, nip, nao)
    assert coul_kpt.shape == (nkpt, nip, nip)
    assert coul_spc.shape == (nspc, nip, nip)

    vks = []
    for dm_kpt in dms:
        rho_kpt = [x @ d @ x.conj().T for x, d in zip(inpv_kpt, dm_kpt)]
        rho_kpt = numpy.asarray(rho_kpt) / nkpt
        assert rho_kpt.shape == (nkpt, nip, nip)

        rho_spc = kpt_to_spc(rho_kpt, phase)
        assert rho_spc.shape == (nspc, nip, nip)
        rho_spc = rho_spc.transpose(0, 2, 1)

        v_spc = coul_spc * rho_spc
        v_spc = numpy.asarray(v_spc).reshape(nspc, nip, nip)

        # v_kpt = phase.T.conj() @ v_spc.reshape(nspc, -1)
        v_kpt = spc_to_kpt(v_spc, phase).conj()
        v_kpt = v_kpt.reshape(nkpt, nip, nip)
        assert v_kpt.shape == (nkpt, nip, nip)

        vks.append([x.conj().T @ v @ x for x, v in zip(inpv_kpt, v_kpt)])

    vks = numpy.asarray(vks).reshape(nset, nkpt, nao, nao)
    if is_zero(kpts_band):
        vks = vks.real
    return _format_jks(vks, dm_kpts, input_band, kpts)