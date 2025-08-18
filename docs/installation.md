# Installation

The nugraph ecosystem consists of two packages, pynuml and nugraph, both of which can be found in the [nugraph repository](https://github.com/nugraph/nugraph). In order to best make use of the pynuml package, it is strongly encouraged to install the provided numl conda environment. The package's parallel processing functionality requires the h5py python module to be installed with MPI support, which cannot be acquired with a simple pip install command.

## Installing dependencies with conda

A conda environment YAML containing all pynuml and nugraph dependencies is available in the main branch of the nugraph repo [here](https://github.com/nugraph/nugraph/blob/main/numl.yaml). This file can be accessed by cloning the nugraph repository, as described below, or downloading it directly by running

```bash
wget https://raw.githubusercontent.com/nugraph/nugraph/refs/heads/main/numl.yaml
```

or

```bash
curl -L -O https://raw.githubusercontent.com/nugraph/nugraph/refs/heads/main/numl.yaml
```

If conda is not already installed, we recommend installing [miniforge](https://github.com/conda-forge/miniforge). Once conda is available, installing the dependencies should be as simple as

```bash
conda env create -f numl.yaml
```

after which the environment can be activated by running

```bash
conda activate numl
```

```{admonition} A note on the conda environment
:class: warning

Note that this environment does not include the pynuml and nugraph packages themselves, only their dependencies. Instructions for installing the packages themselves are provided below.
```

## Installation from source (recommended)

Once dependent packages have been installed, either with conda as described above or otherwise, the pynuml and nugraph packages themselves can be installed from source. The source code can be acquired by running

```bash
git clone git@github.com:nugraph/nugraph
```

Once the repository has been cloned, the nugraph and pynuml packages can be installed into your environment. If you're using a conda environment you installed yourself, the pynuml and nugraph packages can be installed in editable mode using pip. Alternatively, if your conda environment is shared by multiple users, you can enable your local nugraph and pynuml packages for just yourself by prepending them to the `PYTHONPATH` environment variable in your terminal session.

To install your local packages with pip, simply run

```bash
pip install --no-deps -e nugraph/nugraph
pip install --no-deps -e nugraph/pynuml
```

If you're using this method to install your packages, it should only be necessary to run once inside your conda environment. The packages should persist in your conda environment across fresh terminal sessions.

Alternatively, to expose your local packages through the Python path, simply run

```bash
export PYTHONPATH=/path/to/nugraph/nugraph:/path/to/nugraph/pynuml:$PYTHONPATH
```

where `/path/to/nugraph` should be replaced with the absolute path to your local nugraph installation. Note that the setting of this environment variable will only persist for your current terminal session, so you'll need to re-export it whenever you log in, or set it to be automatically exported at login.
