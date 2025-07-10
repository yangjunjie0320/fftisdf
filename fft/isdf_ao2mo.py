import numpy
from functools import reduce
from itertools import product

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.logger import process_clock, perf_counter

import fft
from fft.isdf_jk import kpt_to_spc, spc_to_kpt
from fft.isdf_jk import get_phase_factor

def member(kpt, kpts):
    ind = kpts_helper.member(kpt, kpts)
    assert len(ind) == 1, f"search for {kpt} in {kpts} returns {ind}"
    return ind[0]

def get_ao_eri(df_obj, kpts=None, compact=False): 
    # TODO: compact is not supported yet
    assert compact is False, "compact is not supported"

    log = logger.new_logger(df_obj)

    cell = df_obj.cell
    nao = cell.nao_nr()
    kpts = numpy.asarray(kpts)
    kconserv2 = df_obj.kconserv2
    kconserv3 = df_obj.kconserv3

    k1234 = [member(kpt, df_obj.kpts) for kpt in kpts]
    shape = (nao, nao, nao, nao)
    k1, k2, k3, k4 = k1234
    is_conserved = (kconserv3[k1, k2, k3] == k4)
    if not is_conserved:
        log.warn("kpts are not conserved")
        return numpy.zeros(shape, dtype=numpy.complex128)
    
    # get the Coulomb kernel
    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt
    nkpt, nip, nao = inpv_kpt.shape

    xk1 = inpv_kpt[k1].reshape(nip, -1,  1)
    xk2 = inpv_kpt[k2].reshape(nip,  1, -1)
    xk3 = inpv_kpt[k3].reshape(nip, -1,  1)
    xk4 = inpv_kpt[k4].reshape(nip,  1, -1)

    kq = kconserv2[k1, k2]
    coul_q = coul_kpt[kq]

    rho12 = xk1.conj() * xk2
    rho12 = rho12.reshape(nip, -1)
    vq = lib.dot(rho12.T, coul_q)

    rho34 = xk3.conj() * xk4
    rho34 = rho34.reshape(nip, -1)

    out = lib.dot(vq, rho34)
    out = out.reshape(shape)
    return out

def get_mo_eri(df_obj, mo_coeff_kpts, kpts=None, compact=False):
    # TODO: compact is not supported yet
    assert compact is False, "compact is not supported"

    log = logger.new_logger(df_obj)
    cell = df_obj.cell
    nao = cell.nao_nr()
    kpts = numpy.asarray(kpts)
    kconserv2 = df_obj.kconserv2
    kconserv3 = df_obj.kconserv3

    k1234 = [member(kpt, df_obj.kpts) for kpt in kpts]
    shape = [ck.shape[1] for ck in mo_coeff_kpts]
    k1, k2, k3, k4 = k1234
    is_conserved = (kconserv3[k1, k2, k3] == k4)
    if not is_conserved:
        log.warn("kpts are not conserved")
        return numpy.zeros(shape, dtype=numpy.complex128)
    
    mo_coeff_kpts = [numpy.array(mo_coeff_kpt) for mo_coeff_kpt in mo_coeff_kpts]
    assert len(mo_coeff_kpts) == 4
    
    # get the Coulomb kernel
    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt
    nkpt, nip, nao = inpv_kpt.shape

    xk1 = lib.dot(inpv_kpt[k1], mo_coeff_kpts[0]).reshape(nip, -1,  1)
    xk2 = lib.dot(inpv_kpt[k2], mo_coeff_kpts[1]).reshape(nip,  1, -1)
    xk3 = lib.dot(inpv_kpt[k3], mo_coeff_kpts[2]).reshape(nip, -1,  1)
    xk4 = lib.dot(inpv_kpt[k4], mo_coeff_kpts[3]).reshape(nip,  1, -1)

    kq = kconserv2[k1, k2]
    coul_q = coul_kpt[kq]

    rho12 = xk1.conj() * xk2
    rho12 = rho12.reshape(nip, -1)
    vq = lib.dot(rho12.T, coul_q)

    rho34 = xk3.conj() * xk4
    rho34 = rho34.reshape(nip, -1)

    out = lib.dot(vq, rho34)
    out = out.reshape(shape)
    return out

def ao2mo_7d(df_obj, mo_coeff_kpts, kpts=None, factor=1.0, out=None):
    if kpts is None:
        kpts = df_obj.kpts

    log = logger.new_logger(df_obj)
    t0 = (process_clock(), perf_counter())

    cell = df_obj.cell
    nao = cell.nao_nr()
    kpts = numpy.asarray(kpts)
    assert numpy.all(kpts == df_obj.kpts)
    kconserv2 = df_obj.kconserv2
    kconserv3 = df_obj.kconserv3
    
    phase = get_phase_factor(cell, kpts)
    nspc, nkpt = phase.shape

    kconserv2 = df_obj.kconserv2
    kconserv3 = df_obj.kconserv3

    if isinstance(mo_coeff_kpts, numpy.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts, ] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)
    
    mo_coeff_kpts = [numpy.array(mo_coeff_kpt) for mo_coeff_kpt in mo_coeff_kpts]
    assert len(mo_coeff_kpts) == 4

    # get the Coulomb kernel
    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt
    nip = coul_kpt.shape[1]

    x1_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[0])]
    x1_kpt = numpy.array(x1_kpt)
    n1 = x1_kpt.shape[-1]
    x1_kpt = x1_kpt.reshape(nkpt, nip, n1, 1)

    x2_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[1])]
    x2_kpt = numpy.array(x2_kpt)
    n2 = x2_kpt.shape[-1]
    x2_kpt = x2_kpt.reshape(nkpt, nip, 1, n2)

    x3_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[2])]
    x3_kpt = numpy.array(x3_kpt)
    n3 = x3_kpt.shape[-1]
    x3_kpt = x3_kpt.reshape(nkpt, nip, n3, 1)

    x4_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[3])]
    x4_kpt = numpy.array(x4_kpt)
    n4 = x4_kpt.shape[-1]
    x4_kpt = x4_kpt.reshape(nkpt, nip, 1, n4)

    shape = (nkpt, nkpt, nkpt,) + (n1, n2, n3, n4)
    if out is None:
        eri_7d = numpy.zeros(shape, dtype=numpy.complex128)
    else:
        eri_7d = out
    assert eri_7d.shape == shape

    for k1, k2 in product(range(nkpt), repeat=2):
        kq = kconserv2[k1, k2]
        coul_q = coul_kpt[kq]

        rho12 = x1_kpt[k1].conj() * x2_kpt[k2]
        rho12 = rho12.reshape(nip, n1 * n2)
        vq = lib.dot(rho12.T, coul_q)

        for k3 in range(nkpt):
            k4 = kconserv3[k1, k2, k3]
            rho34 = x3_kpt[k3].conj() * x4_kpt[k4]
            rho34 = rho34.reshape(nip, n3 * n4)
            eri_k1k2k3 = lib.dot(vq, rho34)
            eri_k1k2k3 = eri_k1k2k3.reshape(n1, n2, n3, n4)
            eri_7d[k1, k2, k3] = eri_k1k2k3
            eri_k1k2k3 = None
        
        rho12 = vq = None
    
    log.timer("ao2mo_7d", *t0)
    return eri_7d * factor

def ao2mo_spc(df_obj, mo_coeff_kpts, kpts=None):
    if kpts is None:
        kpts = df_obj.kpts

    log = logger.new_logger(df_obj)

    cell = df_obj.cell
    kpts = numpy.asarray(kpts)
    assert numpy.all(kpts == df_obj.kpts)

    phase = get_phase_factor(cell, kpts)
    nspc, nkpt = phase.shape

    if isinstance(mo_coeff_kpts, numpy.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts, ] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)
    assert len(mo_coeff_kpts) == 4

    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt
    nip = coul_kpt.shape[1]

    t0 = (process_clock(), perf_counter())

    x1_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[0])]
    x1_kpt = numpy.array(x1_kpt)
    x1_spc = kpt_to_spc(x1_kpt, phase)
    n1 = x1_spc.shape[-1]
    x1_spc = x1_spc.reshape(nspc * nip, n1, 1)

    x2_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[1])]
    x2_kpt = numpy.array(x2_kpt)
    x2_spc = kpt_to_spc(x2_kpt, phase)
    n2 = x2_spc.shape[-1]
    x2_spc = x2_spc.reshape(nspc * nip, 1, n2)

    rho12_spc = x1_spc * x2_spc
    rho12_spc = rho12_spc.reshape(nspc, nip, n1 * n2)
    rho12_kpt = spc_to_kpt(rho12_spc, phase)
    rho12_kpt = rho12_kpt.reshape(nkpt, nip, n1 * n2)

    log.timer("prepare rho12", *t0)

    t0 = (process_clock(), perf_counter())

    x3_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[2])]
    x3_kpt = numpy.array(x3_kpt)
    x3_spc = kpt_to_spc(x3_kpt, phase)
    n3 = x3_spc.shape[-1]
    x3_spc = x3_spc.reshape(nspc * nip, n3, 1)

    x4_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[3])]
    x4_kpt = numpy.array(x4_kpt)
    x4_spc = kpt_to_spc(x4_kpt, phase)
    n4 = x4_spc.shape[-1]
    x4_spc = x4_spc.reshape(nspc * nip, 1, n4)

    rho34_spc = x3_spc * x4_spc
    rho34_spc = rho34_spc.reshape(nspc, nip, n3 * n4)
    rho34_kpt = spc_to_kpt(rho34_spc, phase)
    rho34_kpt = rho34_kpt.reshape(nkpt, nip, n3 * n4)

    log.timer("prepare rho34", *t0)

    t0 = (process_clock(), perf_counter())
    shape = (n1, n2, n3, n4)
    eri_spc = numpy.zeros(shape, dtype=numpy.float64)
    for q in range(nkpt):
        coul_q = coul_kpt[q]
        rho12_q = rho12_kpt[q]
        rho34_q = rho34_kpt[q]
        vq = lib.dot(rho12_q.T, coul_q)
        
        eri_q = lib.dot(vq, rho34_q.conj())
        eri_q = eri_q.real * nspc
        eri_q = eri_q.reshape(shape)
        
        eri_spc += eri_q
    log.timer("compute eri_spc", *t0)
    
    return eri_spc

def ao2mo_spc_slow(df_obj, mo_coeff_kpts, kpts=None):
    if kpts is None:
        kpts = df_obj.kpts

    cell = df_obj.cell
    kpts = numpy.asarray(kpts)
    assert numpy.all(kpts == df_obj.kpts)

    phase = get_phase_factor(cell, kpts)
    nspc, nkpt = phase.shape

    if isinstance(mo_coeff_kpts, numpy.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts, ] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)
    assert len(mo_coeff_kpts) == 4

    inpv_kpt = df_obj.inpv_kpt
    coul_kpt = df_obj.coul_kpt
    nip = coul_kpt.shape[1]

    coul_spc = numpy.einsum("kIJ,Rk,Sk->RISJ", coul_kpt, phase.conj(), phase, optimize=True)
    coul_spc = coul_spc.reshape(nspc * nip, nspc * nip).real

    x1_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[0])]
    x1_kpt = numpy.array(x1_kpt)
    x1_spc = kpt_to_spc(x1_kpt, phase)
    n1 = x1_spc.shape[-1]
    x1_spc = x1_spc.reshape(nspc * nip, n1, 1)

    x2_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[1])]
    x2_kpt = numpy.array(x2_kpt)
    x2_spc = kpt_to_spc(x2_kpt, phase)
    n2 = x2_spc.shape[-1]
    x2_spc = x2_spc.reshape(nspc * nip, 1, n2)

    rho12_spc = x1_spc * x2_spc
    rho12_spc = rho12_spc.reshape(nspc * nip, n1 * n2)

    x3_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[2])]
    x3_kpt = numpy.array(x3_kpt)
    x3_spc = kpt_to_spc(x3_kpt, phase)
    n3 = x3_spc.shape[-1]
    x3_spc = x3_spc.reshape(nspc * nip, n3, 1)

    x4_kpt = [lib.dot(xk, ck) for xk, ck in zip(inpv_kpt, mo_coeff_kpts[3])]
    x4_kpt = numpy.array(x4_kpt)
    x4_spc = kpt_to_spc(x4_kpt, phase)
    n4 = x4_spc.shape[-1]
    x4_spc = x4_spc.reshape(nspc * nip, 1, n4)

    rho34_spc = x3_spc * x4_spc
    rho34_spc = rho34_spc.reshape(nspc * nip, n3 * n4)

    eri_spc = reduce(lib.dot, (rho12_spc.T, coul_spc, rho34_spc.conj()))
    return eri_spc.real * nspc
