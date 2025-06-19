import os, unittest
from unittest import TestCase

import numpy, pyscf
from pyscf.pbc.lib.kpts_helper import loop_kkk

class VjkKptTest(TestCase):
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

    def test_krhf_vjk_kpts(self):
        cell = self.cell
        kpts = self.kpts
        tol = self.tol

        krhf = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
        dm0 = krhf.get_init_guess(key="minao")

        vj_ref, vk_ref = self.fftdf_obj.get_jk(dm0, hermi=1, kpts=kpts)
        vj_sol, vk_sol = self.isdf_obj.get_jk(dm0, hermi=1, kpts=kpts)

        err_k = abs(vk_sol - vk_ref).max()
        err_j = abs(vj_sol - vj_ref).max()

        msg = f"Error in vj is {err_j}."
        self.assertLess(err_j, tol, msg)

        msg = f"Error in vk is {err_k}."
        self.assertLess(err_k, tol, msg)

    def test_krhf_vjk_kpts_ewald(self):
        cell = self.cell
        kpts = self.kpts
        tol = self.tol
        
        krhf = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
        dm0 = krhf.get_init_guess(key="minao")
        
        vj_ref, vk_ref = self.fftdf_obj.get_jk(dm0, hermi=1, kpts=kpts, exxdiv="ewald")
        vj_sol, vk_sol = self.isdf_obj.get_jk(dm0, hermi=1, kpts=kpts, exxdiv="ewald")
        
        err_k = abs(vk_sol - vk_ref).max()
        err_j = abs(vj_sol - vj_ref).max()
        
        msg = f"Error in vj is {err_j}."
        self.assertLess(err_j, tol, msg)
        
        msg = f"Error in vk is {err_k}."
        self.assertLess(err_k, tol, msg)

    def test_kuhf_vjk_kpts(self):
        cell = self.cell
        kpts = self.kpts
        tol = self.tol

        kuhf = pyscf.pbc.scf.KUHF(cell, kpts=kpts)
        dm0 = kuhf.get_init_guess(key="minao")

        vj_ref, vk_ref = self.fftdf_obj.get_jk(dm0, hermi=1, kpts=kpts)
        vj_sol, vk_sol = self.isdf_obj.get_jk(dm0, hermi=1, kpts=kpts)

        err_k = abs(vk_sol - vk_ref).max()
        err_j = abs(vj_sol - vj_ref).max()

        msg = f"Error in vj is {err_j}."
        self.assertLess(err_j, tol, msg)

        msg = f"Error in vk is {err_k}."
        self.assertLess(err_k, tol, msg)

    def test_kuhf_vjk_kpts_ewald(self):
        cell = self.cell
        kpts = self.kpts
        tol = self.tol
        
        kuhf = pyscf.pbc.scf.KUHF(cell, kpts=kpts)
        dm0 = kuhf.get_init_guess(key="minao")
        
        vj_ref, vk_ref = self.fftdf_obj.get_jk(dm0, hermi=1, kpts=kpts, exxdiv="ewald")
        vj_sol, vk_sol = self.isdf_obj.get_jk(dm0, hermi=1, kpts=kpts, exxdiv="ewald")
        
        err_k = abs(vk_sol - vk_ref).max()
        err_j = abs(vj_sol - vj_ref).max()
        
        msg = f"Error in vj is {err_j}."
        self.assertLess(err_j, tol, msg)
        
        msg = f"Error in vk is {err_k}."
        self.assertLess(err_k, tol, msg)

if __name__ == "__main__":
    unittest.main()