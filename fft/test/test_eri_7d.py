import numpy
import pyscf
import pyscf.pbc
import pyscf.pbc.gto

class TestERI7D:
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

            err = abs(eri_ao_sol - eri_ao_ref).max()
            assert err < 1e-10

    def test_get_ao_eri_7d(self):
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
        
        import fft
        isdf = fft.ISDF(cell, kpts=kpts)
        isdf.c0 = 20.0
        isdf.tol = 1e-20
        isdf.verbose = 5
        isdf.build()

        # eri_ao_7d_sol = isdf.get_ao_eri_7d(kpts=kpts)

        from pyscf.pbc.lib.kpts_helper import get_kconserv, loop_kkk
        kconserv3 = get_kconserv(cell, kpts)
        for ki, kj, kk in loop_kkk(nkpts):
            km = kconserv3[ki, kj, kk]
            eri_ao_ref = df.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)
            eri_ao_sol = isdf.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)

            err = abs(eri_ao_sol - eri_ao_ref).max()
            print(f"\nki={ki:2d}, kj={kj:2d}, kk={kk:2d}, err={err:6.2e}")            
            assert err < 1e-4


if __name__ == "__main__":
    t = TestERI7D()
    # t.test_fftdf_ao2mo_7d()
    t.test_get_ao_eri_7d()