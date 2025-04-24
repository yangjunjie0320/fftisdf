import unittest

import numpy, pyscf
from pyscf import pbc

import fft

class VjkKptsTest(unittest.TestCase):
    cell = None
    tol = 1e-6

    def setUp(self, kmesh=None):
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

        self.fftdf = pbc.df.FFTDF(cell, kpts=kpts)
        self.isdf  = fft.ISDF(cell, kpts=kpts)
        g0 = cell.gen_uniform_grids(self.cell.mesh)
        inpx = self.isdf.select_inpx(g0=g0, kpts=kpts, tol=1e-30)
        self.isdf.tol = 1e-8
        self.isdf.build(inpx=inpx)

    def test_krhf_vjk_kpts(self):
        cell = self.cell
        kpts = self.kpts
        kmesh = self.kmesh
        tol = self.tol

        krhf = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
        dm0 = krhf.get_init_guess(key="minao")

        vj_ref, vk_ref = self.fftdf.get_jk(dm0, hermi=1, kpts=kpts)
        vj_sol, vk_sol = self.isdf.get_jk(dm0, hermi=1, kpts=kpts)

        err_k = abs(vk_sol - vk_ref).max()
        err_j = abs(vj_sol - vj_ref).max()

        print(f"err_k: {err_k}, err_j: {err_j}")
        print(f"vj_sol: {vj_sol.shape}")
        print(f"vj_ref: {vj_ref.shape}")

        is_vj_close = numpy.allclose(vj_sol, vj_ref, atol=tol)
        self.assertTrue(is_vj_close)

        is_vk_close = numpy.allclose(vk_sol, vk_ref, atol=tol)
        self.assertTrue(is_vk_close)

    def test_kuhf_vjk_kpts(self):
        cell = self.cell
        kpts = self.kpts
        kmesh = self.kmesh
        tol = self.tol

        kuhf = pyscf.pbc.scf.KUHF(cell, kpts=kpts)
        dm0 = kuhf.get_init_guess(key="minao")

        vj_ref, vk_ref = self.fftdf.get_jk(dm0, hermi=1, kpts=kpts)
        vj_sol, vk_sol = self.isdf.get_jk(dm0, hermi=1, kpts=kpts)

        err_k = abs(vk_sol - vk_ref).max()
        err_j = abs(vj_sol - vj_ref).max()

        print(f"err_k: {err_k}, err_j: {err_j}")

        is_vj_close = numpy.allclose(vj_sol, vj_ref, atol=tol)
        self.assertTrue(is_vj_close)

        is_vk_close = numpy.allclose(vk_sol, vk_ref, atol=tol)
        self.assertTrue(is_vk_close)

if __name__ == "__main__":
    unittest.main()