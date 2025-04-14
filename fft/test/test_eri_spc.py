import unittest

import numpy, pyscf
from pyscf import pbc

import fft
import fft.isdf_ao2mo

class EriSpcTest(unittest.TestCase):
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
        cell.basis = "gth-tzvp"
        cell.pseudo = "gth-pbe"
        cell.verbose = 0
        # cell.output = '/dev/null'
        cell.symmetry = False
        cell.build(dump_input=False)

        self.cell = cell
        self.kpts = cell.make_kpts(kmesh)
        self.fftdf = pbc.df.FFTDF(cell, kpts=self.kpts)

        self.isdf  = fft.ISDF(cell, kpts=self.kpts)
        g0 = cell.gen_uniform_grids(cell.mesh)
        self.isdf.inpx = self.isdf.select_inpx(g0=g0, kpts=self.kpts, tol=1e-30)
        self.isdf.tol = 1e-8
        self.isdf.build()

    def test_fftisdf_eri_spc_ao(self):
        cell = self.cell
        kpts = self.kpts
        df_obj = self.fftdf
        isdf_obj = self.isdf

        nkpts = len(kpts)
        nao = cell.nao_nr()

        # coeff_kpts = (numpy.random.random((nkpts, nao, nao)) * 0.9 +
        #               numpy.random.random((nkpts, nao, nao)) * 0.1j)

        coeff_kpts = [numpy.eye(nao) for _ in range(nkpts)]
        coeff_kpts = numpy.array(coeff_kpts)

        eri_spc_ref = isdf_obj.ao2mo_kpt(coeff_kpts, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_spc_ref)
        eri_spc_ref = eri_spc_ref.real

        eri_spc_sol = isdf_obj.ao2mo_spc(coeff_kpts, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(*eri_spc_ref.shape)

        nao2 = nao * nao
        eri_spc_sol = eri_spc_sol.reshape(nao2, nao2)
        eri_spc_ref = eri_spc_ref.reshape(nao2, nao2)

        print("eri_spc_sol = ", eri_spc_sol.shape)
        numpy.savetxt(cell.stdout, eri_spc_sol[:10, :10], fmt="% 8.4f", delimiter=", ")
        print("eri_spc_ref = ", eri_spc_ref.shape)
        numpy.savetxt(cell.stdout, eri_spc_ref[:10, :10], fmt="% 8.4f", delimiter=", ")
        factor = eri_spc_ref / eri_spc_sol
        print("factor = ", factor)
        numpy.savetxt(cell.stdout, factor[:10, :10], fmt="% 8.4f", delimiter=", ")

        is_close = numpy.allclose(eri_spc_sol, eri_spc_ref, atol=1e-6)
        self.assertTrue(is_close)

if __name__ == "__main__":
    # unittest.main()
    test = EriSpcTest()
    test.setUp()
    test.test_fftisdf_eri_spc_ao()