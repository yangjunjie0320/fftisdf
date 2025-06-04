import numpy
from functools import reduce

from pyscf import lib
from itertools import product
from pyscf.pbc.lib import kpts_helper

import fft
from fft.isdf_jk import kpt_to_spc, spc_to_kpt
from fft.isdf_jk import kpts_to_kmesh, get_phase

def member(kpt, kpts):
    ind = kpts_helper.member(kpt, kpts)
    assert len(ind) == 1, f"search for {kpt} in {kpts} returns {ind}"
    return ind[0]

def get_ao_eri(df_obj, kpts=None, compact=False): 
    # TODO: compact is not supported yet
    assert compact is False, "compact is not supported"

    cell = df_obj.cell
    kconserv2 = df_obj.kconserv2
    kconserv3 = df_obj.kconserv3

    km, kn, kl, ks = [member(kpt, df_obj.kpts) for kpt in kpts]
    is_conserved = (kconserv3[km, kn, kl] == ks) # TODO: use kconserv2
    if not is_conserved:
        raise ValueError("kpts are not conserved")
    
    kq = kconserv2[km, kn]
    coul_kpt = df_obj.coul_kpt
    coul_q = coul_kpt[kq]

    inpv_kpt = df_obj._inpv_kpt
    nip, nao = inpv_kpt.shape[1:]
    nao2 = nao * nao # (nao + 1) * nao // 2 if compact

    rho_mn = inpv_kpt[km].conj().reshape(-1, nao, 1) * inpv_kpt[kn].reshape(-1, 1, nao)
    rho_mn = rho_mn.reshape(nip, nao2)

    rho_ls = inpv_kpt[kl].conj().reshape(-1, nao, 1) * inpv_kpt[ks].reshape(-1, 1, nao)
    rho_ls = rho_ls.reshape(nip, nao2)

    eri_ao = reduce(lib.dot, (rho_mn.T, coul_q, rho_ls))
    eri_ao = eri_ao.reshape(nao, nao, nao, nao)
    return eri_ao

def get_mo_eri(df_obj, mo_coeff_kpts, kpts=None, compact=False):
    # TODO: compact is not supported yet
    assert compact is False, "compact is not supported"

    cell = df_obj.cell
    kconserv2 = df_obj.kconserv2
    kconserv3 = df_obj.kconserv3

    km, kn, kl, ks = [member(kpt, df_obj.kpts) for kpt in kpts]
    is_conserved = (kconserv3[km, kn, kl] == ks)
    if not is_conserved:
        raise ValueError("kpts are not conserved")

    kq = kconserv2[km, kn]
    coul_kpt = df_obj.coul_kpt
    coul_q = coul_kpt[kq]
    
    nkpt, nip, nao = df_obj._inpv_kpt.shape

    inpv_m = df_obj._inpv_kpt[km] @ mo_coeff_kpts[0]
    inpv_n = df_obj._inpv_kpt[kn] @ mo_coeff_kpts[1]
    nmo1 = inpv_m.shape[1]
    nmo2 = inpv_n.shape[1]
    rho_mn = inpv_m.conj().reshape(-1, nmo1, 1) * inpv_n.reshape(-1, 1, nmo2)
    rho_mn = rho_mn.reshape(nip, nmo1 * nmo2)

    inpv_l = df_obj._inpv_kpt[kl] @ mo_coeff_kpts[2]
    inpv_s = df_obj._inpv_kpt[ks] @ mo_coeff_kpts[3]
    nmo3 = inpv_l.shape[1]
    nmo4 = inpv_s.shape[1]
    rho_ls = inpv_l.conj().reshape(-1, nmo3, 1) * inpv_s.reshape(-1, 1, nmo4)
    rho_ls = rho_ls.reshape(nip, nmo3 * nmo4)

    eri_mo = reduce(lib.dot, (rho_mn.T, coul_q, rho_ls))
    eri_mo = eri_mo.reshape(nmo1, nmo2, nmo3, nmo4)
    return eri_mo

def ao2mo_7d(df_obj, mo_coeff_kpts, kpts=None):
    # TODO: support different mo_coeff_kpts for each AO index
    if kpts is None:
        kpts = df_obj.kpts

    pcell = df_obj.cell
    kpts = numpy.asarray(kpts)
    assert numpy.all(kpts == df_obj.kpts)

    wrap_around = df_obj.wrap_around
    kpts, kmesh = kpts_to_kmesh(pcell, df_obj.kpts, wrap_around)
    phase = get_phase(pcell, kpts, kmesh, wrap_around)[1]

    # get the Coulomb kernel
    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt
    nkpt, nip, nao = inpv_kpt.shape

    nmo = mo_coeff_kpts.shape[2]
    assert mo_coeff_kpts.shape == (nkpt, nao, nmo)

    nmo2 = nmo * nmo
    shape = (nkpt, ) * 3 + (nmo, ) * 4
    eri_7d = numpy.zeros(shape, dtype=numpy.complex128)

    cell = df_obj.cell
    kconserv2 = df_obj.kconserv2
    kconserv3 = df_obj.kconserv3

    inpv_kpt = df_obj.inpv_kpt @ mo_coeff_kpts
    inpv_kpt = inpv_kpt.reshape(nkpt, nip, nmo)

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
            eri_7d[km, kn, kl] = lib.dot(vq, rho_ls).reshape(nmo, nmo, nmo, nmo)
    
    return eri_7d

def ao2mo_spc(df_obj, mo_coeff_kpts, kpts=None):
    if kpts is None:
        kpts = df_obj.kpts

    pcell = df_obj.cell
    kpts = numpy.asarray(kpts)
    assert numpy.all(kpts == df_obj.kpts)

    wrap_around = df_obj.wrap_around
    kpts, kmesh = kpts_to_kmesh(pcell, df_obj.kpts, wrap_around)
    phase = get_phase(pcell, kpts, kmesh, wrap_around)[1]
    nspc, nkpt = phase.shape

    if isinstance(mo_coeff_kpts, numpy.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts, ] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)

    assert len(mo_coeff_kpts) == 4

    # get the Coulomb kernel
    nkpt, nip, nao = df_obj._inpv_kpt.shape
    coul_kpt = df_obj._coul_kpt

    inpv_kpts = [df_obj._inpv_kpt @ c_kpt for c_kpt in mo_coeff_kpts]
    inpv_kpts = [x_kpt.reshape(nkpt, -1) for x_kpt in inpv_kpts]
    inpv_spcs = [kpt_to_spc(x_kpt, phase) for x_kpt in inpv_kpts]
    inpv_spcs = [x_spc.reshape(nspc, nip, -1) for x_spc in inpv_spcs]

    lhs_spc = inpv_spcs[0].reshape(nspc * nip, -1, 1) * inpv_spcs[1].reshape(nspc * nip, 1, -1)
    lhs_spc = lhs_spc.reshape(nspc, nip, -1)
    lhs_kpt = spc_to_kpt(lhs_spc, phase)
    lhs_kpt = lhs_kpt.reshape(nkpt, nip, -1)
    dl = lhs_kpt.shape[2]
    
    rhs_spc = inpv_spcs[2].reshape(nspc * nip, -1, 1) * inpv_spcs[3].reshape(nspc * nip, 1, -1)
    rhs_spc = rhs_spc.reshape(nspc, nip, -1)
    rhs_kpt = spc_to_kpt(rhs_spc, phase)
    rhs_kpt = rhs_kpt.reshape(nkpt, nip, -1)
    dr = rhs_kpt.shape[2]

    eri_spc = numpy.zeros((dl, dr))
    for q in range(nkpt):
        coul_q = coul_kpt[q]
        lhs_q = lhs_kpt[q].reshape(nip, -1)
        rhs_q = rhs_kpt[q].reshape(nip, -1)

        # eri_q = reduce(lib.dot, (lhs_q.T, coul_q, rhs_q.conj()))
        eri_q = lhs_q.T @ coul_q @ rhs_q.conj()
        eri_q = eri_q.real * nspc
        eri_spc += eri_q
    
    return eri_spc

def ao2mo_spc_slow(df_obj, mo_coeff_kpts, kpts=None):
    if kpts is None:
        kpts = df_obj.kpts

    pcell = df_obj.cell
    kpts = numpy.asarray(kpts)
    assert numpy.all(kpts == df_obj.kpts)

    wrap_around = df_obj.wrap_around
    kpts, kmesh = kpts_to_kmesh(pcell, df_obj.kpts, wrap_around)
    phase = get_phase(pcell, kpts, kmesh, wrap_around)[1]
    nspc, nkpt = phase.shape

    if isinstance(mo_coeff_kpts, numpy.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts, ] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)

    assert len(mo_coeff_kpts) == 4

    # get the Coulomb kernel
    nkpt, nip, nao = df_obj._inpv_kpt.shape
    coul_kpt = df_obj._coul_kpt
    coul_spc = numpy.einsum("kIJ,Rk,Sk->RISJ", coul_kpt, phase.conj(), phase, optimize=True)
    coul_spc = coul_spc.reshape(nspc, nip, nspc, nip).real

    inpv_kpts = [df_obj._inpv_kpt @ c_kpt for c_kpt in mo_coeff_kpts]
    inpv_kpts = [x_kpt.reshape(nkpt, -1) for x_kpt in inpv_kpts]
    inpv_spcs = [kpt_to_spc(x_kpt, phase) for x_kpt in inpv_kpts]
    inpv_spcs = [x_spc.reshape(nspc, nip, -1) for x_spc in inpv_spcs]

    eri_spc = numpy.einsum("RISJ,RIp,RIq,SJr,SJs->pqrs", coul_spc, inpv_spcs[0], inpv_spcs[1], inpv_spcs[2], inpv_spcs[3], optimize=True)
    return eri_spc * nspc

fft.isdf.FFTISDF.get_eri = get_ao_eri
fft.isdf.FFTISDF.get_ao_eri = get_ao_eri
fft.isdf.FFTISDF.get_mo_eri = get_mo_eri
fft.isdf.FFTISDF.ao2mo = get_mo_eri

fft.isdf.FFTISDF.ao2mo_spc = ao2mo_spc
fft.isdf.FFTISDF.ao2mo_kpt = ao2mo_7d
fft.isdf.FFTISDF.ao2mo_7d = ao2mo_7d
