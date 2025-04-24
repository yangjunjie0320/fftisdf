import pyscf
import numpy
from pyscf.pbc.gto import Cell
from pyscf.pbc.df import FFTDF

import fft
from fft.test.test_eri_spc import EriSpcTest
from fft.test.test_eri_kpt import EriKptsTest
from fft.test.test_vjk_kpts import VjkKptsTest

def setup(test_obj, cell=None, basis="gth-dzvp", ke_cutoff=40.0, 
          kmesh=None, isdf_to_save=None, tol=1e-6):
    if kmesh is None:
        kmesh = [2, 2, 2]

    if cell == "diamond-unit-cell":
        cell = Cell()
        cell.atom = ''' 
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = basis
        cell.pseudo = 'gth-pbe'
        cell.a = ''' 
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B' 
        cell.verbose = 5
        cell.ke_cutoff = ke_cutoff
        cell.symmetry = False
        cell.build(dump_input=False)

    elif cell == "he2-cubic-cell":
        cell = Cell()
        cell.atom = '''
        He 0.500000000000   0.500000000000   0.500000000000
        He 0.500000000000   0.500000000000   1.500000000000
        '''
        cell.basis = basis
        cell.pseudo = 'gth-pbe'
        cell.a = '''
        1.000000000, 0.000000000, 0.000000000
        0.000000000, 1.000000000, 0.000000000
        0.000000000, 0.000000000, 2.000000000'''
        cell.ke_cutoff = ke_cutoff
        cell.symmetry = False
        cell.build(dump_input=False)
        
    assert isinstance(cell, Cell)

    kpts = cell.make_kpts(kmesh)
    test_obj.cell = cell
    test_obj.kmesh = kmesh
    test_obj.kpts = kpts
    test_obj.tol = tol

    test_obj.fftdf = FFTDF(cell, kpts=kpts)
    
    test_obj.isdf  = fft.ISDF(cell, kpts=kpts)
    if isdf_to_save is not None:
        test_obj.isdf._isdf = isdf_to_save
        test_obj.isdf._isdf_to_save = isdf_to_save
        inpx = None
    else:
        m0 = cell.cutoff_to_mesh(80.0)
        g0 = cell.gen_uniform_grids(m0)
        inpx = test_obj.isdf.select_inpx(g0=g0, kpts=kpts, tol=1e-30)
        test_obj.isdf.tol = 1e-8
    test_obj.isdf.build(inpx=inpx)

if __name__ == "__main__":
    for kmesh in [[2, 2, 2], [3, 3, 3], [4, 4, 4]]:
        print(f"\nTesting kmesh: {kmesh}")

        kwargs = {
            "basis": "gth-dzvp", "tol": 1e-6,
            "ke_cutoff": 20.0, "kmesh": kmesh,
            "cell": "diamond-unit-cell",
            "isdf_to_save": None, 
        }

        vjk_kpts_test = VjkKptsTest()
        setup(vjk_kpts_test, **kwargs)
        isdf_to_save = vjk_kpts_test.isdf._isdf_to_save
        kwargs["isdf_to_save"] = isdf_to_save

        vjk_kpts_test.test_krhf_vjk_kpts()
        vjk_kpts_test.test_kuhf_vjk_kpts()
        print(f"VjkKptsTest passed for kmesh: {kmesh}")

        eri_kpts_test = EriKptsTest()
        setup(eri_kpts_test, **kwargs)
        eri_kpts_test.test_fftisdf_get_ao_eri()
        print(f"EriKptsTest passed for kmesh: {kmesh}")

        eri_spc_test = EriSpcTest()
        setup(eri_spc_test, **kwargs)
        eri_spc_test.test_fftisdf_eri_spc_mo1()
        eri_spc_test.test_fftisdf_eri_spc_mo4()
        print(f"EriSpcTest passed for kmesh: {kmesh}")
        