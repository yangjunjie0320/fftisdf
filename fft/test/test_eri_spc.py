import os, unittest
from unittest import TestCase

import numpy, pyscf

from fft.isdf_ao2mo import ao2mo_spc_slow
from fft.isdf_jk import get_phase_factor

class EriSpcTest(TestCase):
    cell = None
    tol = 1e-6

    def setUp(self):
        kwargs = {
            "basis": "gth-dzvp", "tol": self.tol,
            "ke_cutoff": 20.0, "kmesh": [1, 1, 3],
            "cell": "he2-cubic-cell", 
            "isdf_to_save": None,
            "output": "/dev/null",
        }

        from fft.test.test_slow import setup
        setup(self, **kwargs)

    def test_fftisdf_eri_spc_slow_mo1(self):
        tol = self.tol
        kpts = self.kpts
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        nmo = nao
        phase = get_phase_factor(self.cell, kpts)

        coeff_spc = numpy.random.random((nkpt, nao, nmo))
        coeff_kpt = numpy.einsum("Rmp,Rk->kmp", coeff_spc, phase)
        coeff_kpt = coeff_kpt.reshape(nkpt, nao, nmo)
        coeff_kpt /= abs(coeff_kpt).max()

        eri_7d = self.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
        eri_spc_ref = eri_spc_ref.reshape(nmo * nmo, nmo * nmo).real

        eri_spc_sol = ao2mo_spc_slow(self.isdf_obj, coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(nmo * nmo, nmo * nmo).real

        err = abs(eri_spc_sol - eri_spc_ref).max()
        msg = f"Error in fftisdf_eri_spc_slow_mo1 is {err}."
        self.assertLess(err, tol, msg)

    def test_fftisdf_eri_spc_slow_mo4(self):
        tol = self.tol
        kpts = self.kpts
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        nmos = [nao // 2, nao, nao // 2, nao]
        phase = get_phase_factor(self.cell, kpts)

        coeff_kpt = []
        for nmo in nmos:
            c_spc = numpy.random.random((nkpt, nao, nmo))
            c_kpt = numpy.einsum("Rmp,Rk->kmp", c_spc, phase)
            c_kpt = c_kpt.reshape(nkpt, nao, nmo)
            c_kpt /= abs(c_kpt).max()
            coeff_kpt.append(c_kpt)

        eri_7d = self.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
        eri_spc_ref = eri_spc_ref.reshape(nmos).real

        eri_spc_sol = ao2mo_spc_slow(self.isdf_obj, coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(nmos).real

        err = abs(eri_spc_sol - eri_spc_ref).max()
        msg = f"Error in fftisdf_eri_spc_slow_mo4 is {err}."
        self.assertLess(err, tol, msg)

    def test_fftisdf_eri_spc_mo1(self):
        tol = self.tol
        kpts = self.kpts
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()
        
        nmo = nao
        phase = get_phase_factor(self.cell, kpts)

        coeff_spc = numpy.random.random((nkpt, nao, nmo))
        coeff_kpt = numpy.einsum("Rmp,Rk->kmp", coeff_spc, phase)
        coeff_kpt = coeff_kpt.reshape(nkpt, nao, nmo)
        coeff_kpt /= abs(coeff_kpt).max()

        eri_7d = self.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
        eri_spc_ref = eri_spc_ref.reshape(nmo * nmo, nmo * nmo).real
        eri_spc_ref /= abs(eri_spc_ref).max()

        eri_spc_sol = self.isdf_obj.ao2mo_spc(coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(nmo * nmo, nmo * nmo).real
        eri_spc_sol /= abs(eri_spc_sol).max()

        err = abs(eri_spc_sol - eri_spc_ref).max()
        msg = f"Error in fftisdf_eri_spc_mo1 is {err}."
        self.assertLess(err, tol, msg)

    def test_fftisdf_eri_spc_mo4(self):
        tol = self.tol
        kpts = self.kpts
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        nmos = [nao // 2, nao, nao // 2, nao]
        phase = get_phase_factor(self.cell, kpts)

        coeff_kpt = []
        for nmo in nmos:
            c_spc = numpy.random.random((nkpt, nao, nmo))
            c_kpt = numpy.einsum("Rmp,Rk->kmp", c_spc, phase)
            c_kpt = c_kpt.reshape(nkpt, nao, nmo)
            c_kpt /= abs(c_kpt).max()
            coeff_kpt.append(c_kpt)

        eri_7d = self.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
        eri_spc_ref = eri_spc_ref.reshape(nmos).real

        eri_spc_sol = self.isdf_obj.ao2mo_spc(coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(nmos).real

        err = abs(eri_spc_sol - eri_spc_ref).max()
        msg = f"Error in fftisdf_eri_spc_mo4 is {err}."
        self.assertLess(err, tol, msg)

if __name__ == "__main__":
    unittest.main()
