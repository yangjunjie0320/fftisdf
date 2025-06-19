import unittest

import numpy, pyscf
from pyscf import pbc
from pyscf.pbc.df import FFTDF

import fft
import fft.isdf_ao2mo

class EriKptsTest(unittest.TestCase):
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
        kpts = cell.make_kpts(kmesh)

        self.cell = cell
        self.kmesh = kmesh
        self.kpts = kpts

        self.fftdf = FFTDF(cell, kpts=kpts)
        self.fftisdf = fft.ISDF(cell, kpts=kpts)
        g0 = cell.gen_uniform_grids(self.cell.mesh)
        inpx = self.fftisdf.select_inpx(g0=g0, kpts=kpts, tol=1e-30)
        self.fftisdf.tol = 1e-8
        self.isdf.build(inpx=inpx)

    def test_fftdf_eri_ao_7d(self):
        """Test for the equivalence of the 7-index ERIs computed by two
        FFTDF member functions.
        """
        cell = self.cell
        kpts = self.kpts
        kmesh = self.kmesh
        tol = self.tol

        nkpts = len(kpts)
        nao = cell.nao_nr()
        coeff_kpts = [numpy.eye(nao) for _ in range(nkpts)]
        coeff_kpts = numpy.array(coeff_kpts)
        coeff_kpts /= abs(coeff_kpts).max()
        eri_ao_7d = self.fftdf.ao2mo_7d(coeff_kpts, kpts=kpts)

        from pyscf.pbc.lib.kpts_helper import loop_kkk
        kconserv3 = self.isdf.kconserv3
        for ki, kj, kk in loop_kkk(nkpts):
            km = kconserv3[ki, kj, kk]

            eri_ao_ref = self.fftdf.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)
            eri_ao_sol = eri_ao_7d[ki, kj, kk]
            eri_ao_sol = eri_ao_sol.reshape(*eri_ao_ref.shape)

            err = abs(eri_ao_sol - eri_ao_ref).max()
            msg = f"Error in eri_ao_7d for kpts {kpts[ki]}, {kpts[kj]}, {kpts[kk]}, {kpts[km]} is {err}."
            self.assertLess(err, tol, msg)

    def test_fftisdf_get_ao_eri(self):
        cell = self.cell
        kpts = self.kpts
        kmesh = self.kmesh
        tol = self.tol

        nkpts = len(kpts)
        nao = cell.nao_nr()

        from pyscf.pbc.lib.kpts_helper import loop_kkk
        kconserv3 = self.isdf.kconserv3
        for ki, kj, kk in loop_kkk(nkpts):
            km = kconserv3[ki, kj, kk]
            eri_ao_ref = self.fftdf.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)
            eri_ao_sol = self.isdf.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]],  compact=False)
            eri_ao_sol = eri_ao_sol.reshape(*eri_ao_ref.shape)

            err = abs(eri_ao_sol - eri_ao_ref).max()
            msg = f"Error in eri_ao_7d for kpts {kpts[ki]}, {kpts[kj]}, {kpts[kk]}, {kpts[km]} is {err}."
            self.assertLess(err, tol, msg)

    def test_fftisdf_eri_ao_7d(self):
        cell = self.cell
        kpts = self.kpts
        kmesh = self.kmesh
        tol = self.tol

        nkpts = len(kpts)
        nao = cell.nao_nr()

        coeff_kpts = [numpy.eye(nao) for _ in range(nkpts)]
        coeff_kpts = numpy.array(coeff_kpts)
        coeff_kpts /= abs(coeff_kpts).max()

        eri_7d_ref = self.fftdf.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = self.isdf.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = eri_7d_sol.reshape(*eri_7d_ref.shape)

        err = abs(eri_7d_sol - eri_7d_ref).max()
        msg = f"Error in eri_ao_7d is {err}."
        self.assertLess(err, tol, msg)

    def test_fftisdf_ao2mo_7d(self):
        cell = self.cell
        kpts = self.kpts
        kmesh = self.kmesh
        tol = self.tol

        nkpts = len(kpts)
        nao = cell.nao_nr()

        coeff_kpts = (numpy.random.random((nkpts, nao, nao)) +
                      numpy.random.random((nkpts, nao, nao)) * 1j)
        coeff_kpts /= abs(coeff_kpts).max()

        eri_7d_ref = self.fftdf.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = self.isdf.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = eri_7d_sol.reshape(*eri_7d_ref.shape)

        err = abs(eri_7d_sol - eri_7d_ref).max()
        msg = f"Error in eri_ao_7d is {err}."
        self.assertLess(err, tol, msg)

if __name__ == "__main__":
    unittest.main()