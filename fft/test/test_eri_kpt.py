import unittest
from unittest import TestCase

import numpy, pyscf
from pyscf import pbc
from pyscf.pbc.df import FFTDF
from pyscf.pbc.lib.kpts_helper import loop_kkk

import fft
import fft.isdf_ao2mo

class EriKptsTest(TestCase):
    cell = None
    tol = 1e-6

    def setUp(self):
        a = 2.0
        lv = numpy.diag([a, a, a * 2])
        atom =  [['He', (0.5 * a, 0.5 * a, 0.5 * a)]]
        atom += [['He', (0.5 * a, 0.5 * a, 1.5 * a)]]

        cell = pyscf.pbc.gto.Cell()
        cell.a = lv
        cell.ke_cutoff = 20.0
        cell.atom = atom
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.verbose = 0
        cell.output = '/dev/null'
        cell.symmetry = False
        cell.build(dump_input=False)

        kmesh = [1, 1, 3]
        wrap_around = True
        kpts = cell.make_kpts(kmesh, wrap_around=wrap_around)

        self.cell = cell
        self.kpts = kpts

        self.fftdf_obj = FFTDF(cell, kpts=kpts)
        self.isdf_obj = fft.ISDF(cell, kpts=kpts)
        self.isdf_obj.build(cisdf=20.0)

    def test_fftdf_eri_ao_7d(self):
        tol = self.tol
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        coeff_kpts = [numpy.eye(nao) for _ in range(nkpt)]
        coeff_kpts = numpy.array(coeff_kpts)
        eri_ao_7d = self.fftdf_obj.ao2mo_7d(coeff_kpts, kpts=self.kpts)
        
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
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        kconserv3 = self.isdf_obj.kconserv3
        for k1, k2, k3 in loop_kkk(nkpt):
            k4 = kconserv3[k1, k2, k3]

            kpts = [self.kpts[k] for k in [k1, k2, k3, k4]]
            eri_ao_ref = self.fftdf_obj.get_ao_eri(kpts, compact=False)
            eri_ao_sol = self.isdf_obj.get_ao_eri(kpts, compact=False)

            eri_ao_ref = eri_ao_ref.reshape(nao, nao, nao, nao)
            eri_ao_sol = eri_ao_sol.reshape(nao, nao, nao, nao)

            err = abs(eri_ao_sol - eri_ao_ref).max()
            msg = f"Error in fftisdf_get_ao_eri for [{k1}, {k2}, {k3}] is {err:6.4e}."
            self.assertLess(err, tol, msg)

    def test_fftisdf_eri_ao_7d(self):
        tol = self.tol
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        coeff_kpts = [numpy.eye(nao) for _ in range(nkpt)]
        coeff_kpts = numpy.array(coeff_kpts)

        eri_7d_ref = self.fftdf_obj.ao2mo_7d(coeff_kpts, kpts=self.kpts)
        eri_7d_sol = self.isdf_obj.ao2mo_7d(coeff_kpts, kpts=self.kpts)

        err = abs(eri_7d_sol - eri_7d_ref).max()
        msg = f"Error in eri_ao_7d is {err:6.4e}."
        self.assertLess(err, tol, msg)

    def test_fftisdf_ao2mo_7d(self):
        tol = self.tol
        nkpt = len(self.kpts)
        nao = self.cell.nao_nr()

        coeff_kpts = (numpy.random.random((nkpt, nao, nao)) +
                      numpy.random.random((nkpt, nao, nao)) * 1j)
        coeff_kpts /= abs(coeff_kpts).max()

        eri_7d_ref = self.fftdf_obj.ao2mo_7d(coeff_kpts, kpts=self.kpts)
        eri_7d_sol = self.isdf_obj.ao2mo_7d(coeff_kpts, kpts=self.kpts)

        err = abs(eri_7d_sol - eri_7d_ref).max()
        msg = f"Error in eri_ao_7d is {err:6.4e}."
        self.assertLess(err, tol, msg)

if __name__ == "__main__":
    unittest.main()
