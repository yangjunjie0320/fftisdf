import unittest

import numpy, pyscf
from pyscf import pbc

import fft
import fft.isdf_ao2mo
from fft.isdf_ao2mo import ao2mo_spc_slow

class EriSpcTest(unittest.TestCase):
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

        self.fftdf = pbc.df.FFTDF(cell, kpts=kpts)
        self.isdf  = fft.ISDF(cell, kpts=kpts)
        g0 = cell.gen_uniform_grids(self.cell.mesh)
        inpx = self.isdf.select_inpx(g0=g0, kpts=kpts, tol=1e-30)
        self.isdf.tol = 1e-8
        self.isdf.build(inpx=inpx)

    def tearDown(self):
        self.cell.stdout.close()

    def test_fft_eri_spc_slow_mo1(self):
        cell = self.cell
        kpts = self.kpts
        isdf_obj = self.isdf

        nkpts = len(kpts)
        nao = cell.nao_nr()
        nmo = nao * 2
        nmo2 = nmo * nmo

        from fft.isdf_jk import get_phase, kpts_to_kmesh
        wrap_around = isdf_obj.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, kpts, wrap_around)
        phase = get_phase(cell, kpts, kmesh, wrap_around)[1]

        coeff_spc = numpy.random.random((nkpts, nao, nmo))
        coeff_kpt = numpy.einsum("Rmp,Rk->kmp", coeff_spc, phase)
        coeff_kpt = coeff_kpt.reshape(nkpts, nao, nmo)

        eri_7d = self.fftdf.ao2mo_7d(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
        eri_spc_ref = eri_spc_ref.reshape(nmo2, nmo2).real

        eri_spc_sol = ao2mo_spc_slow(self.isdf, coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(nmo2, nmo2).real

        is_close = numpy.allclose(eri_spc_sol, eri_spc_ref, atol=self.tol)
        self.assertTrue(is_close)

    def test_fft_eri_spc_slow_mo4(self):
        cell = self.cell
        kpts = self.kpts
        isdf_obj = self.isdf

        nkpts = len(kpts)
        nao = cell.nao_nr()
        nmo = nao * 2
        nmo2 = nmo * nmo

        from fft.isdf_jk import get_phase, kpts_to_kmesh
        wrap_around = isdf_obj.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, kpts, wrap_around)
        phase = get_phase(cell, kpts, kmesh, wrap_around)[1]

        coeff_spc = [numpy.random.random((nkpts, nao, nmo)) for _ in range(4)]
        coeff_kpt = [numpy.einsum("Rmp,Rk->kmp", c, phase) for c in coeff_spc]

        eri_7d = self.fftdf.ao2mo_7d(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
        eri_spc_ref = eri_spc_ref.reshape(nmo2, nmo2).real

        eri_spc_sol = ao2mo_spc_slow(self.isdf, coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(nmo2, nmo2).real

        is_close = numpy.allclose(eri_spc_sol, eri_spc_ref, atol=self.tol)
        self.assertTrue(is_close)

    def test_fftisdf_eri_spc_mo1(self):
        cell = self.cell
        kpts = self.kpts
        isdf_obj = self.isdf

        nkpts = len(kpts)
        nao = cell.nao_nr()
        nmo = nao * 2
        nmo2 = nmo * nmo
        from fft.isdf_jk import get_phase, kpts_to_kmesh
        wrap_around = isdf_obj.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, kpts, wrap_around)
        phase = get_phase(cell, kpts, kmesh, wrap_around)[1]

        coeff_spc = numpy.random.random((nkpts, nao, nmo))
        coeff_kpt = numpy.einsum("Rmp,Rk->kmp", coeff_spc, phase)
        coeff_kpt = coeff_kpt.reshape(nkpts, nao, nmo)

        eri_7d = self.fftdf.ao2mo_7d(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
        eri_spc_ref = eri_spc_ref.reshape(nmo2, nmo2).real

        eri_spc_sol = isdf_obj.ao2mo_spc(coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(nmo2, nmo2).real

        is_close = numpy.allclose(eri_spc_sol, eri_spc_ref, atol=self.tol)
        self.assertTrue(is_close)

    def test_fftisdf_eri_spc_mo4(self):
        cell = self.cell
        kpts = self.kpts
        isdf_obj = self.isdf

        nkpts = len(kpts)
        nao = cell.nao_nr()
        nmo = nao * 2
        nmo2 = nmo * nmo

        from fft.isdf_jk import get_phase, kpts_to_kmesh
        wrap_around = isdf_obj.wrap_around
        kpts, kmesh = kpts_to_kmesh(cell, kpts, wrap_around)
        phase = get_phase(cell, kpts, kmesh, wrap_around)[1]

        coeff_spc = [numpy.random.random((nkpts, nao, nmo)) for _ in range(4)]
        coeff_kpt = [numpy.einsum("Rmp,Rk->kmp", c, phase) for c in coeff_spc]

        eri_7d = self.fftdf.ao2mo_7d(coeff_kpt, kpts=kpts)
        eri_spc_ref = numpy.einsum("KLMmnsl->mnsl", eri_7d)
        eri_spc_ref = eri_spc_ref.reshape(nmo2, nmo2).real

        eri_spc_sol = isdf_obj.ao2mo_spc(coeff_kpt, kpts=kpts)
        eri_spc_sol = eri_spc_sol.reshape(nmo2, nmo2).real

        is_close = numpy.allclose(eri_spc_sol, eri_spc_ref, atol=self.tol)
        self.assertTrue(is_close)

if __name__ == "__main__":
    unittest.main()
