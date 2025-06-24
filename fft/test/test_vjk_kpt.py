import os, unittest
from unittest import TestCase
from fft.test.test_slow import setup

import numpy, pyscf

def krhf_vjk_kpt(test_obj, exxdiv=None):
    cell = test_obj.cell
    kpts = test_obj.kpts
    tol = test_obj.tol
    
    hf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)

    dm0 = hf_obj.get_init_guess(key="minao")
    shape = dm0.shape
    
    vj_ref, vk_ref = test_obj.fftdf_obj.get_jk(dm0, hermi=1, kpts=kpts, exxdiv=exxdiv)
    vj_sol, vk_sol = test_obj.isdf_obj.get_jk(dm0, hermi=1, kpts=kpts, exxdiv=exxdiv)

    err = abs(vj_sol - vj_ref).max()
    msg = f"Error in vj is {err}."
    test_obj.assertLess(err, tol, msg)

    err = abs(vk_sol - vk_ref).max()
    msg = f"Error in vk is {err}."
    test_obj.assertLess(err, tol, msg)

def kuhf_vjk_kpt(test_obj, exxdiv=None):
    cell = test_obj.cell
    kpts = test_obj.kpts
    tol = test_obj.tol
    
    kuhf_obj = pyscf.pbc.scf.KUHF(cell, kpts=kpts)
    dm0 = kuhf_obj.get_init_guess(key="minao")

    vj_ref, vk_ref = test_obj.fftdf_obj.get_jk(dm0, hermi=1, kpts=kpts, exxdiv=exxdiv)
    vj_sol, vk_sol = test_obj.isdf_obj.get_jk(dm0, hermi=1, kpts=kpts, exxdiv=exxdiv)

    err = abs(vj_sol - vj_ref).max()
    msg = f"Error in vj is {err}."
    test_obj.assertLess(err, tol, msg)

    err = abs(vk_sol - vk_ref).max()
    msg = f"Error in vk is {err}."
    test_obj.assertLess(err, tol, msg)

class VjkKptTest(TestCase):
    tol = 1e-6

    def setUp(self):
        self.kwargs = {
            "basis": "gth-dzvp",
            "ke_cutoff": 20.0, 
            "kmesh": [1, 1, 3],
            "cell": "he2-cubic-cell", 
            "isdf_to_save": None,
            "output": "/dev/null", # suppress output
            "verbose": 0, # suppress output
        }

    def test_krhf_vjk_kpt(self):
        kwargs = self.kwargs
        kwargs["wrap_around"] = False
        setup(self, **kwargs)
        krhf_vjk_kpt(self, exxdiv=None)
        krhf_vjk_kpt(self, exxdiv="ewald")

        kwargs["wrap_around"] = True
        setup(self, **kwargs)
        krhf_vjk_kpt(self, exxdiv=None)
        krhf_vjk_kpt(self, exxdiv="ewald")

    def test_kuhf_vjk_kpt(self):
        kwargs = self.kwargs
        kwargs["wrap_around"] = False
        setup(self, **kwargs)
        kuhf_vjk_kpt(self, exxdiv=None)
        kuhf_vjk_kpt(self, exxdiv="ewald")

        kwargs["wrap_around"] = True
        setup(self, **kwargs)
        kuhf_vjk_kpt(self, exxdiv=None)
        kuhf_vjk_kpt(self, exxdiv="ewald")

if __name__ == "__main__":
    unittest.main()