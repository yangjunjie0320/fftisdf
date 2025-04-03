# FFTISDF

## Installation
Use `environment.yml` to create a conda environment.
```bash
conda create -file=environment.yml
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
conda create -file=environment.yml h5py=*=mpi* mpi4ch mpi4py -name=fftisdf-with-mpi
```
