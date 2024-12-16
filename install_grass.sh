#!/bin/bash

# Create and activate conda environment
CONDA_ENV_NAME=grass_gis
#conda create -y -n $CONDA_ENV_NAME python=3.10
conda activate $CONDA_ENV_NAME

# Install required Python packages via conda
: '
conda install -y -c conda-forge \
    numpy \
    six \
    wxpython \
    pandas \
    matplotlib \
    scipy \
    psycopg2 \
    ply \
    termcolor \
    gdal \
    proj \
    geos \
    pdal=1.8.0 \
    python-pdal \
    fftw
'

# Set installation directory in user's home
INSTALL_DIR=$HOME/local/grass-gis
GRASS_VERSION="8.3.1"  # Latest stable version as of April 2024
BUILD_DIR=$HOME/local/grass-build
CONDA_ENV_NAME="grass_gis"

# Create directories
mkdir -p $INSTALL_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Set up local environment for dependencies
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/share/pkgconfig:$PKG_CONFIG_PATH

# Explicitly set PROJ-related environment variables
export PROJ_DIR=$CONDA_PREFIX
export PROJ_INCLUDE=$CONDA_PREFIX/include
export PROJ_LIB=$CONDA_PREFIX/share/proj
export PROJ_CFLAGS="-I$CONDA_PREFIX/include"
export PROJ_LIBS="-L$CONDA_PREFIX/lib -lproj"

export CPPFLAGS="-I$CONDA_PREFIX/include"
export CXXFLAGS="-I$CONDA_PREFIX/include" # needed for pdal

export LD_LIBRARY_PATH=$HOME/local/libiconv/lib:$LD_LIBRARY_PATH
export CPATH=$HOME/local/libiconv/include:$CPATH
export LDFLAGS="-L$HOME/local/libiconv/lib -liconv"
export CPPFLAGS="-I$HOME/local/libinconv/include "${CPPFLAGS}

# Download and compile GRASS GIS
wget https://github.com/OSGeo/grass/archive/refs/tags/${GRASS_VERSION}.tar.gz -O grass-${GRASS_VERSION}.tar.gz
tar xf grass-${GRASS_VERSION}.tar.gz
cd grass-${GRASS_VERSION}

# Configure GRASS GIS with explicit paths
# in the end python wrapper doesn't work
`pwd`/configure \
    --prefix=$INSTALL_DIR \
    --with-gdal=$CONDA_PREFIX/bin/gdal-config \
    --with-proj=$CONDA_PREFIX \
    --with-proj-includes=$CONDA_PREFIX/include \
    --with-proj-libs=$CONDA_PREFIX/lib \
    --with-proj-share=$CONDA_PREFIX/share/proj \
    --with-geos=$CONDA_PREFIX/bin/geos-config \
    --without-cairo \
    --without-postgres \
    --without-mysql \
    --without-sqlite \
    --with-pdal=$CONDA_PREFIX/bin/pdal-config \
    --with-pdal-includes=$CONDA_PREFIX/include \
    --with-fftw-libs=$CONDA_PREFIX/lib \
    --with-fftw-includes=$CONDA_PREFIX/include \
    --without-python
#    --with-python=$CONDA_PREFIX/bin/python3 \
#    --with-python-pkg=grass \

# Fix the wrong relative install-sh path
sed 's:../.././install-sh:/global/u2/y/ywang11/local/grass-build/grass-8.3.1/install-sh:g' include/Make/Platform.make > Platform.make.tmp && mv Platform.make.tmp include/Make/Platform.make

# Compile and install
make >& temp
make install

# Create environment setup script
cat > $INSTALL_DIR/grass-env.sh << EOL
#!/bin/bash

# Activate conda environment
conda activate $CONDA_ENV_NAME

# Set GRASS GIS environment variables
export PATH=$INSTALL_DIR/bin:\$PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH
export GRASSBIN=$INSTALL_DIR/bin/grass
export PYTHONPATH=$INSTALL_DIR/lib/python3/dist-packages:\$PYTHONPATH
export PROJ_LIB=$CONDA_PREFIX/share/proj
EOL

chmod +x $INSTALL_DIR/grass-env.sh

# Create a small test script for PyGRASS
cat > $INSTALL_DIR/test_pygrass.py << 'EOL'
#!/usr/bin/env python3
from grass.pygrass.modules import Module
from grass.script import setup
import os

def test_pygrass():
    # Initialize GRASS GIS session
    gisdb = os.path.join(os.path.expanduser("~"), "grassdata")
    location = "test_location"
    mapset = "PERMANENT"

    # Create necessary directories
    os.makedirs(os.path.join(gisdb, location, mapset), exist_ok=True)

    # Set up GRASS environment
    setup.init(os.environ['GISBASE'], gisdb, location, mapset)

    try:
        # Test PyGRASS by running a simple command
        Module("g.gisenv", flags="s")
        print("PyGRASS test successful!")
    except Exception as e:
        print(f"PyGRASS test failed: {str(e)}")

if __name__ == "__main__":
    test_pygrass()
EOL

chmod +x $INSTALL_DIR/test_pygrass.py

echo "Installation completed. To use GRASS GIS with PyGRASS support, run:"
echo "source $INSTALL_DIR/grass-env.sh"
echo ""
echo "To test PyGRASS installation, run:"
echo "$INSTALL_DIR/test_pygrass.py"