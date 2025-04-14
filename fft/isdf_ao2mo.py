import numpy
from pyscf import lib
from itertools import product
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.pbc.lib.kpts_helper import get_kconserv_ria

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
    kconserv2 = get_kconserv_ria(cell, df_obj.kpts)
    kconserv3 = get_kconserv(cell, df_obj.kpts)

    km, kn, kl, ks = [member(kpt, df_obj.kpts) for kpt in kpts]
    is_conserved = (kconserv3[km, kn, kl] == ks)
    if not is_conserved:
        raise ValueError("kpts are not conserved")
    
    kq = kconserv2[km, kn]
    jq = df_obj._coul_kpt[kq]

    inpv_kpt = df_obj._inpv_kpt
    nip, nao = inpv_kpt.shape[1:]
    nao2 = nao * nao # (nao + 1) * nao // 2 if compact

    rho_mn = inpv_kpt[km].conj().reshape(-1, nao, 1) * inpv_kpt[kn].reshape(-1, 1, nao)
    rho_mn = rho_mn.reshape(nip, nao2)

    rho_ls = inpv_kpt[kl].conj().reshape(-1, nao, 1) * inpv_kpt[ks].reshape(-1, 1, nao)
    rho_ls = rho_ls.reshape(nip, nao2)

    vq = lib.dot(rho_mn.T, jq)
    eri_ao = lib.dot(vq, rho_ls)
    return eri_ao

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
    inpv_kpt = df_obj._inpv_kpt
    coul_kpt = df_obj._coul_kpt
    nkpt, nip, nao = inpv_kpt.shape

    nmo = mo_coeff_kpts.shape[2]
    assert mo_coeff_kpts.shape == (nkpt, nao, nmo)

    nmo2 = nmo * nmo
    shape = (nkpt, ) * 3 + (nmo, ) * 4
    eri_7d = numpy.zeros(shape, dtype=numpy.complex128)

    cell = df_obj.cell
    kconserv2 = get_kconserv_ria(cell, kpts)
    kconserv3 = get_kconserv(cell, kpts)

    # inpv_kpt = numpy.einsum("kIm,kmp->kIp", df_obj._inpv_kpt, mo_coeff_kpts)
    inpv_kpt = df_obj._inpv_kpt @ mo_coeff_kpts
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

    # get the Coulomb kernel
    nkpt, nao, nmo = mo_coeff_kpts.shape
    # inpv_kpt = df_obj._inpv_kpt @ mo_coeff_kpts
    inpv_kpt = numpy.einsum("kIm,kmp->kIp", df_obj._inpv_kpt, mo_coeff_kpts)
    coul_kpt = df_obj._coul_kpt

    nip = inpv_kpt.shape[1]
    # inpv_kpt = inpv_kpt.reshape(nkpt, -1)
    # inpv_spc = kpt_to_spc(inpv_kpt, phase)
    inpv_spc = numpy.einsum("kIp,Rk->RIp", inpv_kpt, phase)
    inpv_spc *= numpy.sqrt(nspc)

    # inpv_spc = inpv_spc.reshape(nspc * nip, nmo)
    # rho_spc = inpv_spc.reshape(-1, nmo, 1) * inpv_spc.reshape(-1, 1, nmo)
    rho_spc = numpy.einsum("RIp,RIq->RIpq", inpv_spc, inpv_spc, optimize=True)
    rho_kpt = numpy.einsum("RIpq,Rk->kIpq", rho_spc, phase, optimize=True)

    # nmo2 = nmo * nmo
    # rho_spc = rho_spc.reshape(nspc, nip, nmo2)
    # rho_kpt = spc_to_kpt(rho_spc, phase)
    # rho_kpt = rho_kpt.reshape(nkpt, nip, nmo2)

    eri_spc = numpy.zeros((nmo, nmo, nmo, nmo))
    for q in range(nkpt):
        rho_q = rho_kpt[q].reshape(nip, -1)
        coul_q = coul_kpt[q]
        v_q = lib.dot(rho_q.T.conj(), coul_q)
        eri_q = lib.dot(v_q, rho_q)
        eri_q = eri_q.real / nspc
        eri_q = eri_q.reshape(nmo, nmo, nmo, nmo)
        eri_spc += eri_q
    return eri_spc

fft.isdf.FFTISDF.get_eri = get_ao_eri
fft.isdf.FFTISDF.get_ao_eri = get_ao_eri
fft.isdf.FFTISDF.ao2mo_spc = ao2mo_spc
fft.isdf.FFTISDF.ao2mo_kpt = ao2mo_7d
fft.isdf.FFTISDF.ao2mo_7d = ao2mo_7d