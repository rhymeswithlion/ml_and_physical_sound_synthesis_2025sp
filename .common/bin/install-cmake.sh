#!/bin/bash

# Installs CMake if it is not already installed.
# Usage: install-cmake.sh

set -e

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "CMake could not be found. Installing CMake..."

    # Determine the OS
    OS=$(uname -s)
    case "$OS" in
        Linux)
            # Install CMake on Linux
            sudo apt-get update
            sudo apt-get install -y cmake
            ;;
        Darwin)
            # Install CMake on macOS
            if ! command -v brew &> /dev/null; then
                echo "Homebrew is not installed. Installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install cmake
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac
else
    echo "CMake is already installed."
fi
