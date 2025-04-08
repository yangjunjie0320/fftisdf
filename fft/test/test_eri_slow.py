import unittest

import numpy, scipy

import pyscf
import pyscf.pbc
import pyscf.pbc.gto

def setUpModule():
    global cell, kpts, df, isdf

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
    cell.build(dump_input=False)

    kpts = cell.make_kpts([4, 4, 4])

    from pyscf.pbc.df import FFTDF
    df = FFTDF(cell, kpts=kpts)

    import fft
    isdf = fft.ISDF(cell, kpts=kpts)
    isdf.tol = 1e-8
    isdf.verbose = 0

    m0 = cell.mesh
    g0 = cell.gen_uniform_grids(m0)
    isdf.inpx = isdf.select_inpx(g0=g0, kpts=kpts, tol=1e-20)
    isdf.build()

def tearDownModule():
    global cell, kpts, df, isdf
    cell.stdout.close()
    del cell, kpts, df, isdf

class KnownValues(unittest.TestCase):
    def test_fftdf_eri_ao_7d(self):
        nkpts = len(kpts)
        nao = cell.nao_nr()

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
            print(f"ki = {ki:2d}, kj = {kj:2d}, kk = {kk:2d}, km = {km:2d}, err = {err:6.2e}")

    def test_fftisdf_get_ao_eri(self):
        nkpts = len(kpts)
        nao = cell.nao_nr()

        from pyscf.pbc.lib.kpts_helper import get_kconserv, loop_kkk
        kconserv3 = get_kconserv(cell, kpts)
        for ki, kj, kk in loop_kkk(nkpts):
            km = kconserv3[ki, kj, kk]
            eri_ao_ref = df.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)
            eri_ao_sol = isdf.get_ao_eri([kpts[ki], kpts[kj], kpts[kk], kpts[km]], compact=False)

            err = abs(eri_ao_sol - eri_ao_ref).max()
            print(f"ki = {ki:2d}, kj = {kj:2d}, kk = {kk:2d}, km = {km:2d}, err = {err:6.2e}")

    def test_fftisdf_eri_ao_7d(self):
        nkpts = len(kpts)
        nao = cell.nao_nr()

        coeff_kpts = [numpy.eye(nao) for _ in range(nkpts)]
        coeff_kpts = numpy.array(coeff_kpts)

        eri_7d_ref = df.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = isdf.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = eri_7d_sol.reshape(*eri_7d_ref.shape)

        err = abs(eri_7d_sol - eri_7d_ref).max()
        print(f"err = {err:6.2e}")

    def test_fftisdf_ao2mo_7d(self):
        nkpts = len(kpts)
        nao = cell.nao_nr()

        coeff_kpts = (numpy.random.random((nkpts, nao, nao)) +
                      numpy.random.random((nkpts, nao, nao)) * 1j)
        eri_7d_ref = df.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = isdf.ao2mo_7d(coeff_kpts, kpts=kpts)
        eri_7d_sol = eri_7d_sol.reshape(*eri_7d_ref.shape)

        err = abs(eri_7d_sol - eri_7d_ref).max()
        print(f"err = {err:6.2e}")

if __name__ == "__main__":
    unittest.main()