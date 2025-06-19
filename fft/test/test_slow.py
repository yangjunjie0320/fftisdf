import os, sys
from pyscf.pbc.gto import Cell
from pyscf.pbc.df import FFTDF

import fft

def setup(test_obj, cell=None, basis="gth-dzvp", ke_cutoff=40.0, 
          kmesh=None, isdf_to_save=None, tol=1e-6, wrap_around=True,
          output=None, verbose=5):
    if kmesh is None:
        kmesh = [2, 2, 2]
    
    if output is None:
        output = "/dev/null"

    if cell == "diamond-unit-cell":
        cell = Cell()
        cell.atom = ''' 
        C 0.0000 0.0000 0.0000
        C 0.8917 0.8917 0.8917
        '''
        cell.a = ''' 
        0.0000 1.7834 1.7834
        1.7834 0.0000 1.7834
        1.7834 1.7834 0.0000
        '''
        cell.unit = 'A' 

    elif cell == "he2-cubic-cell":
        cell = Cell()
        cell.atom = '''
        He 1.0000 1.0000 1.0000
        He 1.0000 1.0000 2.0000
        '''
        cell.a = '''
        2.0000 0.0000 0.0000
        0.0000 2.0000 0.0000
        0.0000 0.0000 3.0000
        '''
        cell.unit = 'A'
    
    cell.verbose = verbose
    cell.ke_cutoff = ke_cutoff
    cell.symmetry = False
    cell.exp_to_discard = 0.1
    cell.basis = basis
    cell.pseudo = 'gth-pbe'
    cell.output = output
    cell.build(dump_input=False)    

    kpts = cell.make_kpts(kmesh, wrap_around)
    test_obj.cell = cell
    test_obj.kmesh = kmesh
    test_obj.kpts = kpts
    test_obj.tol = tol

    test_obj.fftdf_obj = FFTDF(cell, kpts=kpts)
    
    test_obj.isdf_obj  = fft.ISDF(cell, kpts=kpts)
    if isdf_to_save is not None:
        test_obj.isdf_obj._isdf = isdf_to_save
    test_obj.isdf_obj._isdf_to_save = isdf_to_save
    test_obj.isdf_obj.build(cisdf=20.0)

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

    from fft.test.test_vjk_kpts import VjkKptTest
    test_obj = VjkKptTest()
    setup(test_obj, **kwargs)
    test_obj.test_krhf_vjk_kpts()
    test_obj.test_kuhf_vjk_kpts()
    test_obj.test_krhf_vjk_kpts_ewald()
    test_obj.test_kuhf_vjk_kpts_ewald()
    print(f"VjkKptsTest passed for kmesh: {kmesh}")

    from fft.test.test_eri_kpt import EriKptTest
    test_obj = EriKptTest()
    setup(test_obj, **kwargs)
    test_obj.test_fftisdf_get_ao_eri()
    print(f"EriKptsTest passed for kmesh: {kmesh}")

    from fft.test.test_eri_spc import EriSpcTest
    test_obj = EriSpcTest()
    setup(test_obj, **kwargs)
    test_obj.test_fftisdf_eri_spc_mo1()
    test_obj.test_fftisdf_eri_spc_mo4()
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
