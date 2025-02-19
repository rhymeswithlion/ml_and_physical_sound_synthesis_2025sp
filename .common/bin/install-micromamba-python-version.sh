# Installs a specific version of Python using mamba to the provided directory.
# Usage install-mamba-python-version.sh <version> <install_dir>

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: install-micromamba-python-version.sh <version> <install_dir>"
    exit 1
fi

# Check that mamba is installed using the ENV variable MAMBA_EXE
if [ -z "$MAMBA_EXE" ]; then
    echo "MAMBA_EXE could not be found. Please install micromamba and try again."
    exit 1
fi

PYTHON_VERSION=$1
INSTALL_DIR="$2"

# If install directory is relative, make it absolute
if [[ ! "$INSTALL_DIR" = /* ]]; then
    INSTALL_DIR="$(pwd)/$INSTALL_DIR"
fi

# Make sure the install directory is a subdirectory of the current directory
if [[ ! "$INSTALL_DIR" = "$(pwd)"* ]]; then
    echo "Install directory must be a subdirectory of the current directory"
    exit 1
fi

install_python() {
    local PYTHON_VERSION=$1
    local INSTALL_DIR=$2

    # Install Python
    echo "Installing Python $PYTHON_VERSION to $INSTALL_DIR"
    $MAMBA_EXE create -q -y -c conda-forge -n cpython-$PYTHON_VERSION python=$PYTHON_VERSION
    $MAMBA_EXE run -n cpython-$PYTHON_VERSION python -m venv --copies $INSTALL_DIR
    $INSTALL_DIR/bin/python -m pip install --upgrade -q pip wheel
}

# If the correct version of Python is already installed, else install it
if [ -f "$INSTALL_DIR/bin/python" ]; then
    INSTALLED_VERSION=$($INSTALL_DIR/bin/python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
    if [ "$INSTALLED_VERSION" = "$PYTHON_VERSION" ]; then
        echo "Python $PYTHON_VERSION is already installed at $INSTALL_DIR"
    else # Remove the existing Python installation
        rm -rf $INSTALL_DIR
        # Install Python
        install_python $PYTHON_VERSION $INSTALL_DIR
    fi
else
    # Install Python
    install_python $PYTHON_VERSION $INSTALL_DIR
fi

# Install pip, wheel
$INSTALL_DIR/bin/python -m pip install --upgrade -q pip wheel
