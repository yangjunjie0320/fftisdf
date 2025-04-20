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
        cell.output = '/dev/null'
        cell.symmetry = False
        cell.build(dump_input=False)

        self.cell = cell
        self.kpts = cell.make_kpts(kmesh)

        self.isdf  = fft.ISDF(cell, kpts=self.kpts)
        g0 = cell.gen_uniform_grids(cell.mesh)
        self.isdf.inpx = self.isdf.select_inpx(g0=g0, kpts=self.kpts, tol=1e-30)
        self.isdf.tol = 1e-8
        self.isdf.build()

    def tearDown(self):
        self.cell.stdout.close()

    def test_fftisdf_eri_spc_mo1(self):
        cell = self.cell
        kpts = self.kpts
        isdf_obj = self.isdf

        nkpts = len(kpts)
        nao = cell.nao_nr()
        nmo = nao * 2

        from fft.isdf_jk import get_phase, kpts_to_kmesh
        wrap_around = isdf_obj.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, kpts, wrap_around)
        phase = get_phase(cell, kpts, kmesh, wrap_around)[1]

        coeff_spc = numpy.random.random((nkpts, nao, nmo))
        coeff_kpt = numpy.einsum("Rmp,Rk->Rmp", coeff_spc, phase)
        coeff_kpt = coeff_kpt.reshape(nkpts, nao, nmo)

        eri_spc_ref = isdf_obj.ao2mo_kpt(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_spc_ref)
        eri_spc_ref = eri_spc_ref.real

        eri_spc_sol = isdf_obj.ao2mo_spc(coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(*eri_spc_ref.shape)

        is_close = numpy.allclose(eri_spc_sol, eri_spc_ref, atol=1e-10)
        self.assertTrue(is_close)

    def test_fftisdf_eri_spc_mo4(self):
        cell = self.cell
        kpts = self.kpts
        isdf_obj = self.isdf

        nkpts = len(kpts)
        nao = cell.nao_nr()
        nmo = nao * 2

        from fft.isdf_jk import get_phase, kpts_to_kmesh
        wrap_around = isdf_obj.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, kpts, wrap_around)
        phase = get_phase(cell, kpts, kmesh, wrap_around)[1]

        coeff_spc = [numpy.random.random((nkpts, nao, nmo)) for _ in range(4)]
        coeff_kpt = [numpy.einsum("Rmp,Rk->Rmp", c, phase) for c in coeff_spc]

        eri_spc_ref = isdf_obj.ao2mo_kpt(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_spc_ref)
        eri_spc_ref = eri_spc_ref.real

        eri_spc_sol = isdf_obj.ao2mo_spc(coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(*eri_spc_ref.shape)

        is_close = numpy.allclose(eri_spc_sol, eri_spc_ref, atol=1e-10)
        self.assertTrue(is_close)

if __name__ == "__main__":
    unittest.main()
