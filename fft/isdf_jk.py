import os, sys
import numpy, scipy

import pyscf
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero

from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools.k2gamma import get_phase
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 2000))

def kpts_to_kmesh(cell, kpts, wrap_around=False):
    if not isinstance(kpts, numpy.ndarray):
        kpts = numpy.asarray(kpts.kpts)
    kmesh = pbctools.k2gamma.kpts_to_kmesh(cell, kpts - kpts[0])
    assert numpy.allclose(kpts, cell.get_kpts(kmesh, wrap_around=wrap_around))
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

def get_k_kpts(df_obj, dm_kpts, hermi=1, kpts=numpy.zeros((1, 3)), kpts_band=None,
               exxdiv=None):
    cell = df_obj.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension == 3
    assert hermi == 1

    cell = df_obj.cell
    kpts = numpy.asarray(kpts)
    assert numpy.all(kpts == df_obj.kpts)

    wrap_around = df_obj.wrap_around
    kpts, kmesh = kpts_to_kmesh(cell, df_obj.kpts, wrap_around)
    phase = get_phase(cell, kpts, kmesh, wrap_around)[1]

    nao = cell.nao_nr()
    nkpt = nspc = numpy.prod(kmesh)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpt, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == nkpt, "not supporting kpts_band"

    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt

    nip = inpv_kpt.shape[1]
    coul_spc = kpt_to_spc(coul_kpt, phase)
    coul_spc = coul_spc * numpy.sqrt(nkpt)
    coul_spc = coul_spc.reshape(nspc, nip, nip)
    
    assert inpv_kpt.shape == (nkpt, nip, nao)
    assert coul_kpt.shape == (nkpt, nip, nip)
    assert coul_spc.shape == (nspc, nip, nip)

    vk_kpts = []
    for dm_kpt in dms:
        rho_kpt = inpv_kpt @ dm_kpt @ inpv_kpt.conj().transpose(0, 2, 1)
        rho_kpt = numpy.asarray(rho_kpt) / nkpt
        assert rho_kpt.shape == (nkpt, nip, nip)

        rho_spc = kpt_to_spc(rho_kpt, phase)
        assert rho_spc.shape == (nspc, nip, nip)
        rho_spc = rho_spc.transpose(0, 2, 1)

        v_spc = coul_spc * rho_spc

        v_kpt = spc_to_kpt(v_spc, phase)
        v_kpt = v_kpt.reshape(nkpt, nip, nip)

        # vk_kpt = numpy.einsum("kIJ,kIm,kJn->kmn", v_kpt, inpv_kpt.conj(), inpv_kpt, optimize=True)
        vk_kpt = inpv_kpt.transpose(0, 2, 1) @ v_kpt @ inpv_kpt.conj()
        vk_kpt = vk_kpt.conj().reshape(nkpt, nao, nao)
        vk_kpts.append(vk_kpt)

    vk_kpts = numpy.asarray(vk_kpts).reshape(nset, nkpt, nao, nao)
    if is_zero(kpts_band):
        vk_kpts = vk_kpts.real

    if exxdiv is not None:
        from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
        assert exxdiv.lower() == "ewald"
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)
