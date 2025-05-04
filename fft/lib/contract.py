import numpy, ctypes, time, os

def contract_v1(f_kpt, g_kpt, phase):
    """Contract two k-space arrays."""
    nk, m, n = f_kpt.shape
    ns, nk = phase.shape
    l = g_kpt.shape[1]

    assert f_kpt.shape == (nk, m, n)
    assert g_kpt.shape == (nk, l, n)

    # kmn,kln -> kml
    t_kpt = f_kpt.conj() @ g_kpt.transpose(0, 2, 1)
    t_kpt = t_kpt.reshape(nk, m, l)
    
    # kpt -> spc
    t_spc = phase @ t_kpt.reshape(nk, m * l)
    t_spc = t_spc.real  # .reshape(ns, m, l)

    # smn,smn -> smn
    x_spc = t_spc * t_spc
    x_spc = x_spc

    # spc -> kpt
    x_kpt = phase.conj().T @ x_spc.reshape(ns, m * l)
    x_kpt = x_kpt.reshape(nk, m, l)
    return x_kpt

def contract_v2(f_kpt, g_kpt, phase):
    """Contract two k-space arrays."""
    nk, m, n = f_kpt.shape
    ns, nk = phase.shape
    l = g_kpt.shape[1]

    assert f_kpt.shape == (nk, m, n)
    assert g_kpt.shape == (nk, l, n)

    print(f_kpt.shape, g_kpt.shape, phase.shape)
    print(f_kpt.dtype, g_kpt.dtype, phase.dtype)
    
    # compile the C code with mkl and openmp
    """
    On macos:
    gcc -O3 -shared -lopenblas -fopenmp -o libfftisdf.dylib -fPIC contract.c -I$(brew --prefix openblas)/include -L$(brew --prefix openblas)/lib
    On linux: 
    gcc -O3 -shared \
    -I/home/junjiey/anaconda3/envs/fftisdf/include \
    -L/home/junjiey/anaconda3/envs/fftisdf/lib \
    -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -fopenmp \
    -o libfftisdf.so -fPIC contract.c
    """
    libfftisdf = numpy.ctypeslib.load_library('libfftisdf', os.path.dirname(__file__))
    func = getattr(libfftisdf, 'contract', None)
    assert func is not None

    print(f"nk = {nk}")
    print(f"phase = {phase}")

    f_kpt_conj = f_kpt.conj()
    x_kpt = numpy.zeros((nk, m, l), dtype=numpy.complex128)
    func(
        x_kpt.ctypes.data_as(ctypes.c_void_p),
        f_kpt_conj.ctypes.data_as(ctypes.c_void_p),
        g_kpt.ctypes.data_as(ctypes.c_void_p),
        phase.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nk), ctypes.c_int(m), 
        ctypes.c_int(n), ctypes.c_int(l)
    )
    return x_kpt

contract = contract_v2

if __name__ == "__main__":
    for nk in [1, 5, 10]:
        ns = nk
        m = 100
        n = 200
        l = 50

        f_kpt = numpy.random.rand(nk, m, n) + 1j * numpy.random.rand(nk, m, n)
        g_kpt = numpy.random.rand(nk, l, n) + 1j * numpy.random.rand(nk, l, n)
        phase = numpy.random.rand(ns, nk) + 1j * numpy.random.rand(ns, nk)

        t0 = time.time()
        x_kpt_ref = contract_v1(f_kpt, g_kpt, phase)
        t_ref = time.time() - t0

        t0 = time.time()
        x_kpt_sol = contract_v2(f_kpt, g_kpt, phase)
        t_sol = time.time() - t0

        err = abs(x_kpt_sol - x_kpt_ref).max()
        print("err = %6.2e, t_ref = %6.2e, t_sol = %6.2e" % (err, t_ref, t_sol))
