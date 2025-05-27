import os, sys
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
        C 0.0000 0.0000 0.0000
        C 0.8917 0.8917 0.8917
        '''
        cell.basis = basis
        cell.pseudo = 'gth-pbe'
        cell.a = ''' 
        0.0000 1.7834 1.7834
        1.7834 0.0000 1.7834
        1.7834 1.7834 0.0000
        '''
        cell.unit = 'A' 
        cell.verbose = 5
        cell.ke_cutoff = ke_cutoff
        cell.symmetry = False
        cell.exp_to_discard = 0.1
        cell.build(dump_input=False)

    elif cell == "he2-cubic-cell":
        cell = Cell()
        cell.atom = '''
        He 1.0000 1.0000 1.0000
        He 1.0000 1.0000 2.0000
        '''
        cell.basis = basis
        cell.pseudo = 'gth-pbe'
        cell.a = '''
        2.0000 0.0000 0.0000
        0.0000 2.0000 0.0000
        0.0000 0.0000 3.0000
        '''
        cell.unit = 'A'
        cell.verbose = 5
        cell.ke_cutoff = ke_cutoff
        cell.symmetry = False
        cell.exp_to_discard = 0.1
        cell.build(dump_input=False)
        
    assert isinstance(cell, Cell)

    kpts = cell.make_kpts(kmesh)
    test_obj.cell = cell
    test_obj.kmesh = kmesh
    test_obj.kpts = kpts
    test_obj.tol = tol

    test_obj.fftdf = FFTDF(cell, kpts=kpts)
    
    test_obj.isdf  = fft.ISDF(cell, kpts=kpts)
    if os.path.exists(isdf_to_save):
        test_obj.isdf._isdf = isdf_to_save
        inpx = None
    else:
        g0 = cell.gen_uniform_grids(cell.mesh)
        inpx = test_obj.isdf.select_inpx(g0=g0, c0=None, kpts=kpts, tol=1e-30)
    test_obj.isdf.tol = 1e-8
    test_obj.isdf._isdf_to_save = isdf_to_save
    test_obj.isdf.build(inpx=inpx)

def main(cell="diamond-unit-cell", kmesh=None, ke_cutoff=20.0, tol=1e-6):
    if kmesh is None:
        kmesh = [2, 2, 2]

    kwargs = {
        "basis": "gth-dzvp", "tol": tol,
        "ke_cutoff": ke_cutoff, "kmesh": kmesh,
        "cell": cell, "isdf_to_save": "isdf.h5",
    }

    if os.path.exists(kwargs["isdf_to_save"]):
        os.remove(kwargs["isdf_to_save"])

    vjk_kpts_test = VjkKptsTest()
    setup(vjk_kpts_test, **kwargs)
    vjk_kpts_test.test_krhf_vjk_kpts()
    vjk_kpts_test.test_kuhf_vjk_kpts()
    vjk_kpts_test.test_krhf_vjk_kpts_ewald()
    vjk_kpts_test.test_kuhf_vjk_kpts_ewald()
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
    
if __name__ == "__main__":
    for kmesh in [[2, 2, 2], [3, 3, 3], [4, 4, 4]]:
        for cell in ["he2-cubic-cell", "diamond-unit-cell"]:
            msg = f"testing {cell} with kmesh: {kmesh}, ke_cutoff: 20.0, tol: 1e-6\n"
            print("\n\nStart %s" % msg)
            try:
                main(cell=cell, kmesh=kmesh, ke_cutoff=20.0, tol=1e-5)
                print("Passed %s\n\n" % msg)
            except Exception as e:
                print(e)
                print("Failed %s\n\n" % msg)
