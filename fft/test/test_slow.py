import os, sys, numpy, scipy
from pyscf.pbc.gto import Cell
from pyscf.pbc.df import FFTDF

import fft

def setup(test_obj, cell=None, basis="gth-dzvp", ke_cutoff=40.0, 
          kmesh=None, tol=1e-6, wrap_around=True, output=None, 
          cisdf=20.0, verbose=5):
    if kmesh is None:
        kmesh = [2, 2, 2]

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

    elif cell == "h2o-box":
        l = 5.0
        box = numpy.eye(3) * l

        atom = []
        atom.append(['O', [l / 2, l / 2, l / 2]])
        atom.append(['H', [l / 2 - 0.689440, l / 2 + 0.578509, l / 2]])
        atom.append(['H', [l / 2 + 0.689440, l / 2 - 0.578509, l / 2]])

        cell = Cell()
        cell.atom = atom
        cell.a = box
        cell.unit = 'B'

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

    assert cell is not None
    
    cell.verbose = verbose
    cell.ke_cutoff = ke_cutoff
    cell.symmetry = False
    cell.exp_to_discard = 0.1
    cell.basis = basis
    cell.pseudo = 'gth-pbe'
    if output is not None:
        cell.output = output
    cell.build(dump_input=False)

    kpts = cell.make_kpts(kmesh, wrap_around)

    test_obj.cell = cell
    test_obj.kmesh = kmesh
    test_obj.kpts = kpts
    test_obj.tol = tol

    test_obj.fftdf_obj = FFTDF(cell, kpts=kpts)
    
    test_obj.isdf_obj  = fft.ISDF(cell, kpts=kpts)
    test_obj.isdf_obj.verbose = verbose
    test_obj.isdf_obj.build(cisdf=cisdf)

def main(kwargs):
    from fft.test.test_vjk_kpt import VjkKptTest
    test_obj = VjkKptTest()
    setup(test_obj, **kwargs)
    
    from fft.test.test_vjk_kpt import krhf_vjk_kpt
    from fft.test.test_vjk_kpt import kuhf_vjk_kpt
    krhf_vjk_kpt(test_obj)
    kuhf_vjk_kpt(test_obj)
    print(f"VjkKptsTest passed for kmesh: {kmesh}")

    from fft.test.test_eri_kpt import fftisdf_get_ao_eri
    fftisdf_get_ao_eri(test_obj)
    print(f"EriKptsTest passed for kmesh: {kmesh}")

    from fft.test.test_eri_spc import fftisdf_eri_spc_mo1
    from fft.test.test_eri_spc import fftisdf_eri_spc_mo4
    fftisdf_eri_spc_mo1(test_obj)
    fftisdf_eri_spc_mo4(test_obj)
    print(f"EriSpcTest passed for kmesh: {kmesh}")

if __name__ == "__main__":
    kwargs = {
        "ke_cutoff": 40.0,
        "tol": 1e-6,
        "verbose": 0,
        "basis": "gth-dzvp",
        "cisdf": 20.0,
        "verbose": 6
    }
    
    from itertools import product
    cells = ["he2-cubic-cell", "diamond-unit-cell", "h2o-box"]
    kmeshes = [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
    wrap_arounds = [True, False]
    loop = product(cells, kmeshes, wrap_arounds)
    
    for cell, kmesh, wrap_around in loop:
        kwargs["cell"] = cell
        kwargs["kmesh"] = kmesh
        kwargs["wrap_around"] = wrap_around
        msg = f"testing {cell} with kmesh: {kmesh}, wrap_around: {wrap_around}"
        print("\n\nStart %s" % msg)
        try:
            main(kwargs)
            print("Passed %s\n\n" % msg)
        except Exception as e:
            print(e)
            print("Failed %s\n\n" % msg)
