import os, unittest
from unittest import TestCase
from fft.test.test_slow import setup

import numpy, pyscf

def fftisdf_get_ao_eri(test_obj):
    tol = test_obj.tol
    kpts = test_obj.kpts
    nkpt = len(test_obj.kpts)
    nao = test_obj.cell.nao_nr()

    kconserv3 = test_obj.isdf_obj.kconserv3

    from pyscf.pbc.lib.kpts_helper import loop_kkk
    for k1, k2, k3 in loop_kkk(nkpt):
        k4 = kconserv3[k1, k2, k3]

        k1234 = kpts[[k1, k2, k3, k4]]
        eri_ao_ref = test_obj.fftdf_obj.get_ao_eri(k1234, compact=False)
        eri_ao_sol = test_obj.isdf_obj.get_ao_eri(k1234, compact=False)

        eri_ao_ref = eri_ao_ref.reshape(nao, nao, nao, nao)
        eri_ao_sol = eri_ao_sol.reshape(nao, nao, nao, nao)

        err = abs(eri_ao_sol - eri_ao_ref).max()
        msg = f"Error in fftisdf_get_ao_eri for [{k1}, {k2}, {k3}] is {err:6.4e}."
        test_obj.assertLess(err, tol, msg)

def fftisdf_eri_ao_7d(test_obj):
    tol = test_obj.tol
    kpts = test_obj.kpts
    nkpt = len(test_obj.kpts)
    nao = test_obj.cell.nao_nr()

    coeff_kpt = [numpy.eye(nao) for _ in range(nkpt)]
    coeff_kpt = numpy.array(coeff_kpt)

    eri_7d_ref = test_obj.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
    eri_7d_sol = test_obj.isdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)

    err = abs(eri_7d_sol - eri_7d_ref).max()
    msg = f"Error in eri_ao_7d is {err:6.4e}."
    test_obj.assertLess(err, tol, msg)

def fftisdf_ao2mo_7d(test_obj):
    tol = test_obj.tol
    kpts = test_obj.kpts
    nkpt = len(test_obj.kpts)
    nao = test_obj.cell.nao_nr()

    nmo = nao
    coeff_kpt  = numpy.random.random((nkpt, nao, nmo)) * 1j
    coeff_kpt += numpy.random.random((nkpt, nao, nmo))
    norm = numpy.linalg.norm(coeff_kpt, axis=1)
    coeff_kpt /= norm[:, None]
    assert norm.shape == (nkpt, nmo)

    eri_7d_ref = test_obj.fftdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)
    eri_7d_sol = test_obj.isdf_obj.ao2mo_7d(coeff_kpt, kpts=kpts)

    err = abs(eri_7d_sol - eri_7d_ref).max()
    msg = f"Error in eri_ao_7d is {err:6.4e}."
    test_obj.assertLess(err, tol, msg)

class EriKptTest(TestCase):
    tol = 1e-6

    def setUp(self):
        self.kwargs = {
            "basis": "gth-dzvp",
            "ke_cutoff": 30.0, 
            "kmesh": [1, 1, 3],
            "cell": "he2-cubic-cell", 
            "output": "/dev/null", # suppress output
            "verbose": 0, # suppress output
        }

    def test_fftisdf_get_ao_eri(self):
        kwargs = self.kwargs
        kwargs["wrap_around"] = False
        setup(self, **kwargs)
        fftisdf_get_ao_eri(self)
        
        kwargs["wrap_around"] = True
        setup(self, **kwargs)
        fftisdf_get_ao_eri(self)

    def test_fftisdf_eri_ao_7d(self):
        kwargs = self.kwargs
        kwargs["wrap_around"] = False
        setup(self, **kwargs)
        fftisdf_eri_ao_7d(self)

        kwargs["wrap_around"] = True
        setup(self, **kwargs)
        fftisdf_eri_ao_7d(self)

    def test_fftisdf_ao2mo_7d(self):
        kwargs = self.kwargs
        kwargs["wrap_around"] = False
        setup(self, **kwargs)
        fftisdf_ao2mo_7d(self)

        kwargs["wrap_around"] = True
        setup(self, **kwargs)
        fftisdf_ao2mo_7d(self)

if __name__ == "__main__":
    unittest.main()
