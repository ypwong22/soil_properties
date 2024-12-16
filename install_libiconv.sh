# Download libiconv (1.17 is the latest stable version)
cd ~/local
wget https://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.17.tar.gz
tar xzf libiconv-1.17.tar.gz
cd libiconv-1.17

# Configure and install to your local directory
./configure --prefix=$HOME/local/libiconv
make
make install

export LD_LIBRARY_PATH=$HOME/local/libiconv/lib:$LD_LIBRARY_PATH
export CPATH=$HOME/local/libiconv/include:$CPATH