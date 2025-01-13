Installation
============

In order to best make use of the **pynuml** package, it is strongly encouraged to install the provided numl Anaconda environment. Parallel processing functionality requires an MPI installation, which will be automatically configured when you install the `numl` conda environment.

Installing the numl conda environment
-------------------------------------

Installing **pynuml** requires an Anaconda installation that utilises `conda-forge`. If you need to install Anaconda, we recommend using the `Mambaforge`_ variant.

A conda environment for numl is available via the anaconda client, and can be installed using::

    mamba install -y anaconda-client
    mamba env create numl/numl

Once installed, this environment will need to be activated at the start of each terminal session::

    mamba activate numl

.. _Mambaforge: https://github.com/conda-forge/miniforge#mambaforge

This environment contains the most recent version of **pynuml** published to conda.

Installing with Anaconda
------------------------

It is also possible to install **pynuml** on its own via Anaconda, using the **numl** channel::

    mamba install -c numl pynuml

Installing with pip
-------------------

**pynuml** is also available on PyPi, although this installation method is not recommended, as **pynuml** has non-python dependencies that cannot be installed by pip. If the user has installed those dependencies manually, then the package can be installed using::

    pip install pynuml

Installing for development
--------------------------

If you're installing **pynuml** for development, you can install the numl Anaconda environment as outlined above, and then clone the repository directly and install it in editable mode::

    git clone https://github.com/nugraph/pynuml
    pip install --no-deps -e ./pynuml

This will uninstall the conda release of pynuml installed by default as part of the numl environment, and override it with your local repository. If installed in editable mode, any changes made to the package will instantaneously be reflected when the module is imported in Python.
