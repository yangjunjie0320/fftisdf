import unittest

import numpy, scipy

import pyscf
import pyscf.pbc
import pyscf.pbc.gto

class TestERI7D(unittest.TestCase):
    def test_fftdf_ao2mo_7d(self):
        a = 3.0
        lv = numpy.diag([a, a, a * 2])
        atom = [['He', (0.5 * a, 0.5 * a, 0.5 * a)]]
        atom += [['He', (0.5 * a, 0.5 * a, 1.5 * a)]]

        cell = pyscf.pbc.gto.Cell()
        cell.a = lv
        cell.ke_cutoff = 10.0
        cell.atom = atom
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.verbose = 0
        cell.build(dump_input=False)

        kpts = cell.make_kpts([1, 1, 3])
        nkpts = len(kpts)
        nao = cell.nao_nr()

        from pyscf.pbc.df import FFTDF
        df = FFTDF(cell, kpts=kpts)
        df.build()

        coeff_kpts = [numpy.eye(nao) for _ in range(nkpts)]
        coeff_kpts = numpy.array(coeff_kpts)
        eri_ao_7d_sol = df.ao2mo_7d(coeff_kpts, kpts=kpts)

        from pyscf.pbc.lib.kpts_helper import get_kconserv, loop_kkk
        kconserv3 = get_kconserv(cell, kpts)

        for ki, kj, kk in loop_kkk(nkpts):
            km = kconserv3[ki, kj, kk]
            eri_ao_ref = df.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)

            eri_ao_sol = eri_ao_7d_sol[ki, kj, kk]
            eri_ao_sol = eri_ao_sol.reshape(*eri_ao_ref.shape)
            is_close = numpy.allclose(eri_ao_sol, eri_ao_ref, atol=1e-10)
            self.assertTrue(is_close)

    def test_get_ao_eri_7d(self):
        a = 3.0
        lv = numpy.diag([a, a, a * 2])
        atom = [['He', (0.5 * a, 0.5 * a, 0.5 * a)]]
        atom += [['He', (0.5 * a, 0.5 * a, 1.5 * a)]]

        cell = pyscf.pbc.gto.Cell()
        cell.a = lv
        cell.ke_cutoff = 10.0
        cell.atom = atom
        cell.basis = "gth-tzvp"
        cell.pseudo = "gth-pbe"
        cell.verbose = 0
        cell.build(dump_input=False)

        kpts = cell.make_kpts([3, 3, 3])
        nkpts = len(kpts)
        nao = cell.nao_nr()

        from pyscf.pbc.df import FFTDF
        df = FFTDF(cell, kpts=kpts)
        df.build()
        
        import fft
        isdf = fft.ISDF(cell, kpts=kpts)
        isdf.tol = 1e-10
        isdf.verbose = 5

        k0 = 10.0
        m0 = cell.cutoff_to_mesh(k0)
        g0 = cell.gen_uniform_grids(m0)
        print(f"m0 = {m0}, g0.shape = {g0.shape}")
        inpx = fft.isdf.select_inpx(isdf, g0=g0, c0=None, tol=1e-16, kpts=kpts)
        print(f"inpx.shape = {inpx.shape}")
        isdf.inpx = inpx
        isdf.build()

        # eri_ao_7d_sol = isdf.get_ao_eri_7d(kpts=kpts)

        from pyscf.pbc.lib.kpts_helper import get_kconserv, loop_kkk
        kconserv3 = get_kconserv(cell, kpts)
        for ki, kj, kk in loop_kkk(nkpts):
            km = kconserv3[ki, kj, kk]
            eri_ao_ref = df.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)
            eri_ao_sol = isdf.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)

            err = abs(eri_ao_sol - eri_ao_ref).max()
            print(f"ki={ki}, kj={kj}, kk={kk}, km={km}, err = {err:6.2e}")
            is_close = err < 1e-4
            self.assertTrue(is_close)

if __name__ == "__main__":
    t = TestERI7D()
    # t.test_fftdf_ao2mo_7d()
    t.test_get_ao_eri_7d()