#!/bin/bash
# Installation script for Linux/macOS

# Check Python version
python_version=$(python3 --version | awk '{print $2}')
required_version="3.10.0"

# Function to compare versions
version_compare() {
    python3 -c "from packaging import version; print(version.parse('$1') < version.parse('$2'))"
}

if [ "$(version_compare "$python_version" "$required_version")" = "True" ]; then
    echo "Error: Python 3.10 or higher is required"
    echo "Current Python version: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Run the Python installation script
echo "Running installation script..."
python install.py --with-samples

echo "Installation complete. Activate the virtual environment with:"
echo "source venv/bin/activate"
