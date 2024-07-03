#!/bin/bash
git clone https://github.com/NU-CUCIS/ph5concat
cd ph5concat
autoreconf -i
./configure --prefix=$CONDA_PREFIX \
            --with-mpi=$CONDA_PREFIX \
            --with-hdf5=$CONDA_PREFIX \
            CFLAGS="-O2 -DNDEBUG" \
            CXXFLAGS="-O2 -DNDEBUG" \
            LIBS="-ldl -lz" \
            --enable-profiling
make install
cd ..
rm -fr ph5concat
