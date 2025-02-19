#!/bin/bash

# Installs Ruby if it is not already installed.
# Usage: install-ruby.sh

set -e

# Check if Ruby is installed
if ! command -v ruby &> /dev/null; then
    echo "Ruby could not be found. Installing Ruby..."

    # Determine the OS
    OS=$(uname -s)
    case "$OS" in
        Linux)
            # Install Ruby on Linux
            sudo apt-get update
            sudo apt-get install -y ruby
            ;;
        Darwin)
            # Install Ruby on macOS
            if ! command -v brew &> /dev/null; then
                echo "Homebrew is not installed. Installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install ruby
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac
else
    echo "Ruby is already installed."
fi
