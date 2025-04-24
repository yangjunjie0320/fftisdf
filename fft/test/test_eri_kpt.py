import unittest

import numpy, pyscf
from pyscf import pbc

import fft
import fft.isdf_ao2mo

class EriKptsTest(unittest.TestCase):
    cell = None

    def setUp(self, kmesh=None):
        if kmesh is None:
            kmesh = [1, 1, 3]
        
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

        self.cell = cell
        self.kpts = cell.make_kpts(kmesh)

        self.fftdf = pbc.df.FFTDF(cell, kpts=self.kpts)
        self.isdf  = fft.ISDF(cell, kpts=self.kpts)
        g0 = cell.gen_uniform_grids(self.cell.mesh)
        inpx = self.isdf.select_inpx(g0=g0, kpts=self.kpts, tol=1e-30)
        self.isdf.tol = 1e-8
        self.isdf.build(inpx=inpx)

    def test_fftdf_eri_ao_7d(self):
        """Test for the equivalence of the 7-index ERIs computed by two
        FFTDF member functions.
        """
        cell = self.cell
        kpts = self.kpts
        df_obj = self.fftdf

        nkpts = len(kpts)
        nao = cell.nao_nr()
        coeff_kpts = [numpy.eye(nao) for _ in range(nkpts)]
        coeff_kpts = numpy.array(coeff_kpts)
        eri_ao_7d = df_obj.ao2mo_7d(coeff_kpts, kpts=kpts)

        from pyscf.pbc.lib.kpts_helper import get_kconserv, loop_kkk
        kconserv3 = get_kconserv(cell, kpts)

        for ki, kj, kk in loop_kkk(nkpts):
            km = kconserv3[ki, kj, kk]

            eri_ao_ref = df_obj.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)
            eri_ao_sol = eri_ao_7d[ki, kj, kk]
            eri_ao_sol = eri_ao_sol.reshape(*eri_ao_ref.shape)

            is_close = numpy.allclose(eri_ao_sol, eri_ao_ref, atol=1e-6)
            self.assertTrue(is_close)

    def test_fftisdf_get_ao_eri(self):
        cell = self.cell
        kpts = self.kpts
        df_obj = self.fftdf

        nkpts = len(kpts)
        nao = cell.nao_nr()

        from pyscf.pbc.lib.kpts_helper import get_kconserv, loop_kkk
        kconserv3 = get_kconserv(cell, kpts)
        for ki, kj, kk in loop_kkk(nkpts):
            km = kconserv3[ki, kj, kk]
            eri_ao_ref = self.fftdf.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)
            eri_ao_sol = self.isdf.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]],  compact=False)

            is_close = numpy.allclose(eri_ao_sol, eri_ao_ref, atol=1e-6)
            self.assertTrue(is_close)

    def test_fftisdf_eri_ao_7d(self):
        cell = self.cell
        kpts = self.kpts
        df_obj = self.fftdf
        isdf_obj = self.isdf

        nkpts = len(kpts)
        nao = cell.nao_nr()

        coeff_kpts = [numpy.eye(nao) for _ in range(nkpts)]
        coeff_kpts = numpy.array(coeff_kpts)

        eri_7d_ref = df_obj.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = isdf_obj.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = eri_7d_sol.reshape(*eri_7d_ref.shape)

        is_close = numpy.allclose(eri_7d_sol, eri_7d_ref, atol=1e-6)
        self.assertTrue(is_close)

    def test_fftisdf_ao2mo_7d(self):
        cell = self.cell
        kpts = self.kpts
        df_obj = self.fftdf
        isdf_obj = self.isdf

        nkpts = len(kpts)
        nao = cell.nao_nr()

        coeff_kpts = (numpy.random.random((nkpts, nao, nao)) +
                      numpy.random.random((nkpts, nao, nao)) * 1j)
        eri_7d_ref = df_obj.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = isdf_obj.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = eri_7d_sol.reshape(*eri_7d_ref.shape)

        is_close = numpy.allclose(eri_7d_sol, eri_7d_ref, atol=1e-6)
        self.assertTrue(is_close)

if __name__ == "__main__":
    # unittest.main()
    test = EriKptsTest()
    test.test_fftisdf_ao2mo_7d()