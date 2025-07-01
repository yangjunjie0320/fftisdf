import os, unittest
from unittest import TestCase
from fft.test.test_slow import setup

import numpy, pyscf

def fftisdf_eri_spc_slow_mo1(test_obj):
    tol = test_obj.tol
    kpts = test_obj.kpts
    nkpt = len(test_obj.kpts)
    nao = test_obj.cell.nao_nr()

    from fft.isdf_jk import get_phase_factor
    phase = get_phase_factor(test_obj.cell, kpts)

    nmo = nao
    coeff_spc = numpy.random.random((nkpt, nao, nmo))
    coeff_kpt = numpy.einsum("Rmp,Rk->kmp", coeff_spc, phase)
    coeff_kpt = coeff_kpt.reshape(nkpt, nao, nmo)
    norm = numpy.linalg.norm(coeff_kpt, axis=2)
    coeff_kpt /= norm[:, :, None]

    eri_7d = test_obj.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
    eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
    eri_spc_ref = eri_spc_ref.reshape(nmo * nmo, nmo * nmo).real

    from fft.isdf_ao2mo import ao2mo_spc_slow
    eri_spc_sol = ao2mo_spc_slow(test_obj.isdf_obj, coeff_kpt, kpts=kpts)
    eri_spc_sol = eri_spc_sol.reshape(nmo * nmo, nmo * nmo).real

    err = abs(eri_spc_sol - eri_spc_ref).max()
    msg = f"Error in fftisdf_eri_spc_slow_mo1 is {err}."
    test_obj.assertLess(err, tol, msg)

def fftisdf_eri_spc_slow_mo4(test_obj):
    tol = test_obj.tol
    kpts = test_obj.kpts
    nkpt = len(test_obj.kpts)
    nao = test_obj.cell.nao_nr()

    from fft.isdf_jk import get_phase_factor
    phase = get_phase_factor(test_obj.cell, kpts)

    coeff_kpt = []
    nmos = [nao // 2, nao, nao // 2, nao]
    for nmo in nmos:
        c_spc = numpy.random.random((nkpt, nao, nmo))
        c_kpt = numpy.einsum("Rmp,Rk->kmp", c_spc, phase)
        c_kpt = c_kpt.reshape(nkpt, nao, nmo)
        norm = numpy.linalg.norm(c_kpt, axis=2)
        c_kpt /= norm[:, :, None]
        coeff_kpt.append(c_kpt)

    eri_7d = test_obj.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
    eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
    eri_spc_ref = eri_spc_ref.reshape(nmos).real

    from fft.isdf_ao2mo import ao2mo_spc_slow
    eri_spc_sol = ao2mo_spc_slow(test_obj.isdf_obj, coeff_kpt, kpts=kpts)
    eri_spc_sol = eri_spc_sol.reshape(nmos).real

    err = abs(eri_spc_sol - eri_spc_ref).max()
    msg = f"Error in fftisdf_eri_spc_slow_mo4 is {err}."
    test_obj.assertLess(err, tol, msg)

def fftisdf_eri_spc_mo1(test_obj):
    tol = test_obj.tol
    kpts = test_obj.kpts
    nkpt = len(test_obj.kpts)
    nao = test_obj.cell.nao_nr()    

    from fft.isdf_jk import get_phase_factor
    phase = get_phase_factor(test_obj.cell, kpts)
    
    nmo = nao
    coeff_spc = numpy.random.random((nkpt, nao, nmo))
    coeff_kpt = numpy.einsum("Rmp,Rk->kmp", coeff_spc, phase)
    coeff_kpt = coeff_kpt.reshape(nkpt, nao, nmo)
    norm = numpy.linalg.norm(coeff_kpt, axis=2)
    coeff_kpt /= norm[:, :, None]

    eri_7d = test_obj.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
    eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
    eri_spc_ref = eri_spc_ref.reshape(nmo * nmo, nmo * nmo).real

    eri_spc_sol = test_obj.isdf_obj.ao2mo_spc(coeff_kpt, kpts=kpts)
    eri_spc_sol = eri_spc_sol.reshape(nmo * nmo, nmo * nmo).real

    err = abs(eri_spc_sol - eri_spc_ref).max()
    msg = f"Error in fftisdf_eri_spc_mo1 is {err}."
    test_obj.assertLess(err, tol, msg)

def fftisdf_eri_spc_mo4(test_obj):
    tol = test_obj.tol
    kpts = test_obj.kpts
    nkpt = len(test_obj.kpts)
    nao = test_obj.cell.nao_nr()
    nmos = [nao // 2, nao, nao // 2, nao]

    from fft.isdf_jk import get_phase_factor
    phase = get_phase_factor(test_obj.cell, kpts)

    coeff_kpt = []
    for nmo in nmos:
        c_spc = numpy.random.random((nkpt, nao, nmo))
        c_kpt = numpy.einsum("Rmp,Rk->kmp", c_spc, phase)
        c_kpt = c_kpt.reshape(nkpt, nao, nmo)
        norm = numpy.linalg.norm(c_kpt, axis=2)
        c_kpt /= norm[:, :, None]
        coeff_kpt.append(c_kpt)

    eri_7d = test_obj.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
    eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
    eri_spc_ref = eri_spc_ref.reshape(nmos).real

    eri_spc_sol = test_obj.isdf_obj.ao2mo_spc(coeff_kpt, kpts=kpts)
    eri_spc_sol = eri_spc_sol.reshape(nmos).real

    err = abs(eri_spc_sol - eri_spc_ref).max()
    msg = f"Error in fftisdf_eri_spc_mo4 is {err}."
    test_obj.assertLess(err, tol, msg)

class EriSpcTest(TestCase):
    tol = 1e-6

    def setUp(self):
        self.kwargs = {
            "basis": "gth-dzvp",
            "ke_cutoff": 20.0, 
            "kmesh": [1, 1, 3],
            "cell": "he2-cubic-cell", 
            "output": "/dev/null", # suppress output
            "verbose": 0, # suppress output
        }

    def test_fftisdf_eri_spc_slow_mo1(self):
        kwargs = self.kwargs
        kwargs["wrap_around"] = False
        setup(self, **kwargs)
        fftisdf_eri_spc_slow_mo1(self)
        
        kwargs["wrap_around"] = True
        setup(self, **kwargs)
        fftisdf_eri_spc_slow_mo1(self)

    def test_fftisdf_eri_spc_slow_mo4(self):
        kwargs = self.kwargs
        kwargs["wrap_around"] = False
        setup(self, **kwargs)
        fftisdf_eri_spc_slow_mo4(self)

        kwargs["wrap_around"] = True
        setup(self, **kwargs)
        fftisdf_eri_spc_slow_mo4(self)

    def test_fftisdf_eri_spc_mo1(self):
        kwargs = self.kwargs
        kwargs["wrap_around"] = False
        setup(self, **kwargs)
        fftisdf_eri_spc_mo1(self)

        kwargs["wrap_around"] = True
        setup(self, **kwargs)
        fftisdf_eri_spc_mo1(self)

    def test_fftisdf_eri_spc_mo4(self):
        kwargs = self.kwargs
        kwargs["wrap_around"] = False
        setup(self, **kwargs)
        fftisdf_eri_spc_mo4(self)


if __name__ == "__main__":
    unittest.main()
