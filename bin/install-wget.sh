#!/bin/bash

# Installs wget if it is not already installed.
# Usage: install-wget.sh

set -e

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo "wget could not be found. Installing wget..."

    # Determine the OS
    OS=$(uname -s)
    case "$OS" in
        Linux)
            # Install wget on Linux
            sudo apt-get update
            sudo apt-get install -y wget
            ;;
        Darwin)
            # Install wget on macOS
            if ! command -v brew &> /dev/null; then
                echo "Homebrew is not installed. Installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install wget
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac
else
    echo "wget is already installed."
fi
