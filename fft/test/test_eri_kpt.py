import os, unittest
from unittest import TestCase

import numpy, pyscf

from pyscf.pbc.lib.kpts_helper import loop_kkk

class EriKptTest(TestCase):
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

    def test_fftdf_eri_ao_7d(self):
        tol = self.tol
        kpts = self.kpts
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        coeff_kpt = [numpy.eye(nao) for _ in range(nkpt)]
        coeff_kpt = numpy.array(coeff_kpt)
        eri_ao_7d = self.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
        
        kconserv3 = self.isdf_obj.kconserv3
        for k1, k2, k3 in loop_kkk(nkpt):
            k4 = kconserv3[k1, k2, k3]

            kpts = [self.kpts[k] for k in [k1, k2, k3, k4]]
            eri_ao_ref = self.fftdf_obj.get_ao_eri(kpts, compact=False)
            eri_ao_sol = eri_ao_7d[k1, k2, k3]

            eri_ao_ref = eri_ao_ref.reshape(nao, nao, nao, nao)
            eri_ao_sol = eri_ao_sol.reshape(nao, nao, nao, nao)

            err = abs(eri_ao_sol - eri_ao_ref).max()
            msg = f"Error in fftdf_eri_ao_7d for kpts [{k1}, {k2}, {k3}] is {err:6.4e}."
            self.assertLess(err, tol, msg)

    def test_fftisdf_get_ao_eri(self):
        tol = self.tol
        kpts = self.kpts
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        kconserv3 = self.isdf_obj.kconserv3
        for k1, k2, k3 in loop_kkk(nkpt):
            k4 = kconserv3[k1, k2, k3]

            k1234 = kpts[[k1, k2, k3, k4]]
            eri_ao_ref = self.fftdf_obj.get_ao_eri(k1234, compact=False)
            eri_ao_sol = self.isdf_obj.get_ao_eri(k1234, compact=False)

            eri_ao_ref = eri_ao_ref.reshape(nao, nao, nao, nao)
            eri_ao_sol = eri_ao_sol.reshape(nao, nao, nao, nao)

            err = abs(eri_ao_sol - eri_ao_ref).max()
            msg = f"Error in fftisdf_get_ao_eri for [{k1}, {k2}, {k3}] is {err:6.4e}."
            self.assertLess(err, tol, msg)

    def test_fftisdf_eri_ao_7d(self):
        tol = self.tol
        kpts = self.kpts
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        coeff_kpt = [numpy.eye(nao) for _ in range(nkpt)]
        coeff_kpt = numpy.array(coeff_kpt)

        eri_7d_ref = self.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
        eri_7d_sol = self.isdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)

        err = abs(eri_7d_sol - eri_7d_ref).max()
        msg = f"Error in eri_ao_7d is {err:6.4e}."
        self.assertLess(err, tol, msg)

    def test_fftisdf_ao2mo_7d(self):
        tol = self.tol
        kpts = self.kpts
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        coeff_kpts = (numpy.random.random((nkpt, nao, nao)) +
                      numpy.random.random((nkpt, nao, nao)) * 1j)
        coeff_kpts /= abs(coeff_kpts).max()

        eri_7d_ref = self.fftdf_obj.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = self.isdf_obj.ao2mo_7d(coeff_kpts, kpts=kpts)

        err = abs(eri_7d_sol - eri_7d_ref).max()
        msg = f"Error in eri_ao_7d is {err:6.4e}."
        self.assertLess(err, tol, msg)

if __name__ == "__main__":
    unittest.main()
