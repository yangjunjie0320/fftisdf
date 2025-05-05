# FFTISDF

## Installation

Use `environment.yml` to create a conda environment.

```bash
# make it strict channel priority
conda env create --file=environment.yml --name=fftisdf
conda activate fftisdf
```

Then activate the `conda` environment, add the current directory to the `PYTHONPATH`.

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

Test the installation by running the following command.

```bash
python -c "import fft; from fft import ISDF"
```

If you want to use the MPI version, install the following dependencies
in the same conda environment.

```bash
conda env create --file=environment.yml --name=fftisdf-with-mpi -y
conda activate fftisdf-with-mpi
conda install h5py=*=mpi* mpich mpi4py -y
```

## Examples

The following examples are provided to test the installation. For hybrid
density functional theory calculations,

```bash
python examples/01-diamond-rks.py
```

## Test
The unit tests are provided in the `fft/test` directory. Install `pytest`
before running the tests.

```bash
pip install pytest
```

Then run the tests.

```bash
pytest
```
# fftisdf-for-dmet
