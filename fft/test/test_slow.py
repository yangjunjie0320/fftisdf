import unittest

from fft.test.test_eri_kpts import EriKptsTest
from fft.test.test_vjk_kpts import VjkKptsTest

if __name__ == "__main__":
    for kmesh in [[2, 2, 2], [3, 3, 3], [4, 4, 4]]:
        print(f"Testing kmesh: {kmesh}")

        eri_kpts_test = EriKptsTest()
        eri_kpts_test.setUp(kmesh)
        eri_kpts_test.test_fftdf_eri_ao_7d()
        eri_kpts_test.test_fftisdf_get_ao_eri()
        eri_kpts_test.test_fftisdf_eri_ao_7d()
        eri_kpts_test.test_fftisdf_ao2mo_7d()
        print(f"EriKptsTest passed for kmesh: {kmesh}")

        vjk_kpts_test = VjkKptsTest()
        vjk_kpts_test.setUp(kmesh)
        vjk_kpts_test.test_krhf_vjk_kpts()
        vjk_kpts_test.test_kuhf_vjk_kpts()
        print(f"VjkKptsTest passed for kmesh: {kmesh}")
        