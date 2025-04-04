# FFTISDF

## Installation

Use `environment.yml` to create a conda environment.

```bash
# make it strict channel priority
conda env create --file=environment.yml --name=fftisdf
conda activate fftisdf
```

Then activate the conda environment, add the current directory to the PYTHONPATH.

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

Test the installation by running the following command.

```bash
python -c "import fft; from fft import FFTISDF"
```

If you want to use the MPI version, install the following dependencies
in the same conda environment.

```bash
conda env create --file=environment.yml --name=fftisdf-with-mpi -y
conda activate fftisdf-with-mpi
conda install h5py=*=mpi* mpich mpi4py -y
```
