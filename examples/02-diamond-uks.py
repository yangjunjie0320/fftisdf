import numpy, pyscf
from pyscf.pbc import gto, dft

import fft

a = 1.7834
lv    = numpy.ones((3, 3)) * a
lv   -= numpy.diag([a, a, a])
atom  = [("C", [0.00000,  0.00000,  0.00000])]
atom += [("C", [0.5 * a,  0.5 * a,  0.5 * a])]

cell = gto.Cell()
cell.unit = 'A'
cell.atom = atom
cell.a = lv
cell.basis = "gth-dzvp"
cell.pseudo = "gth-pbe"
cell.ke_cutoff = 200.0
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 2, 2])
mf = dft.KUKS(cell, kpts)
mf.xc = "PBE0"
mf.conv_tol = 1e-6
mf.max_cycle = 50
mf.verbose = 4
mf.exxdiv = None
mf.with_df = fft.ISDF(cell, kpts)
mf.with_df.c0 = 10.0
mf.with_df.verbose = 5
mf.with_df.build()
mf.kernel()

e_tot = mf.e_tot
print("FFTDF    e_tot = -10.93142766")
print("FFT-ISDF e_tot = %12.8f" % e_tot)
