import unittest

from fft.test.test_eri_spc import EriSpcTest
from fft.test.test_eri_kpt import EriKptsTest
from fft.test.test_vjk_kpts import VjkKptsTest

# def setup_diamond_unit_cell(test_obj, kmesh):
#     a = 2.0
#     lv = numpy.diag([a, a, a])
#     atom = [['C', (0.0, 0.0, 0.0)], ['C', (0.5 * a, 0.5 * a, 0.5 * a)]]
#     cell = pyscf.pbc.gto.Cell()
#     cell.a = lv
#     cell.atom = atom

if __name__ == "__main__":
    for kmesh in [[2, 2, 2], [3, 3, 3], [4, 4, 4]]:
        print(f"Testing kmesh: {kmesh}")

        vjk_kpts_test = VjkKptsTest()
        vjk_kpts_test.setUp(kmesh)
        vjk_kpts_test.test_krhf_vjk_kpts()
        vjk_kpts_test.test_kuhf_vjk_kpts()
        print(f"VjkKptsTest passed for kmesh: {kmesh}")

        eri_kpts_test = EriKptsTest()
        eri_kpts_test.setUp(kmesh)
        eri_kpts_test.test_fftisdf_get_ao_eri()
        print(f"EriKptsTest passed for kmesh: {kmesh}")

        eri_spc_test = EriSpcTest()
        eri_spc_test.setUp(kmesh)
        # eri_spc_test.test_fft_eri_spc_slow_mo1()
        # eri_spc_test.test_fft_eri_spc_slow_mo4()
        eri_spc_test.test_fftisdf_eri_spc_mo1()
        eri_spc_test.test_fftisdf_eri_spc_mo4()
        print(f"EriSpcTest passed for kmesh: {kmesh}")
        