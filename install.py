#!/usr/bin/env python3
"""
Installation script for crowpeas.

This script helps users set up crowpeas with uv, handling the PyTorch
installation first (which is important for GPU support), and then installing
crowpeas and its dependencies.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def check_uv_installed():
    """Check if uv is installed and install it if not."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✓ uv is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing uv package manager...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
            print("✓ uv installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install uv. Please install it manually with: pip install uv")
            return False


def detect_cuda():
    """Detect if CUDA is available on the system."""
    # Check for nvidia-smi
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        print("✓ CUDA detected via nvidia-smi")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # On Windows, check for CUDA path
        if platform.system() == "Windows":
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path and os.path.exists(cuda_path):
                print(f"✓ CUDA detected at {cuda_path}")
                return True
        
        print("ℹ No CUDA detected, will install CPU version of PyTorch")
        return False


def setup_venv():
    """Create a virtual environment using uv."""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print(f"ℹ Virtual environment already exists at {venv_path}")
        use_existing = input("Use existing environment? [Y/n]: ").strip().lower()
        if use_existing in ('', 'y', 'yes'):
            return str(venv_path)
        else:
            print(f"Creating new virtual environment...")
    else:
        print(f"Creating virtual environment at {venv_path}...")
    
    try:
        subprocess.run(["uv", "venv", str(venv_path)], check=True)
        print(f"✓ Virtual environment created at {venv_path}")
        return str(venv_path)
    except subprocess.CalledProcessError:
        print(f"❌ Failed to create virtual environment at {venv_path}")
        sys.exit(1)


def install_pytorch(venv_path, cuda_available):
    """Install PyTorch with the appropriate CUDA support."""
    if cuda_available:
        print("Installing PyTorch with CUDA support...")
        pytorch_cmd = ["uv", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
    else:
        print("Installing PyTorch (CPU version)...")
        pytorch_cmd = ["uv", "pip", "install", "torch", "torchvision", "torchaudio"]
    
    activate_venv = get_activate_command(venv_path)
    
    try:
        if platform.system() == "Windows":
            cmd = f"{activate_venv} && {' '.join(pytorch_cmd)}"
            subprocess.run(cmd, shell=True, check=True)
        else:
            cmd = f"source {activate_venv} && {' '.join(pytorch_cmd)}"
            subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
        print("✓ PyTorch installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install PyTorch")
        sys.exit(1)


def install_crowpeas(venv_path):
    """Install crowpeas in development mode."""
    print("Installing crowpeas and dependencies...")
    
    activate_venv = get_activate_command(venv_path)
    
    try:
        if platform.system() == "Windows":
            cmd = f"{activate_venv} && uv pip install -e .[dev]"
            subprocess.run(cmd, shell=True, check=True)
        else:
            cmd = f"source {activate_venv} && uv pip install -e .[dev]"
            subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
        print("✓ crowpeas installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install crowpeas")
        sys.exit(1)


def get_activate_command(venv_path):
    """Get the appropriate activation command for the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "activate")
    else:
        return os.path.join(venv_path, "bin", "activate")


def print_activation_instructions(venv_path):
    """Print instructions for activating the virtual environment."""
    if platform.system() == "Windows":
        activate_cmd = f"{venv_path}\\Scripts\\activate"
    else:
        activate_cmd = f"source {venv_path}/bin/activate"
    
    print("\n" + "=" * 80)
    print("Installation completed successfully!")
    print("=" * 80)
    print(f"\nTo activate the virtual environment, run:")
    print(f"  {activate_cmd}")
    print("\nTo test if crowpeas is installed correctly, run:")
    print("  crowpeas --help")
    print("\nTo generate a sample configuration, run:")
    print("  crowpeas -g")
    print("\nTo run crowpeas with a configuration file, run:")
    print("  crowpeas -d -t -v config.toml")
    print("\nTo enable PyTorch model compilation (requires C++ compiler):")
    print("  export CROWPEAS_COMPILE=1")
    print("\nFor more information, see the documentation at:")
    print("  https://crowpeas.readthedocs.io/\n")


def main():
    print("=" * 80)
    print("crowpeas Installer")
    print("=" * 80)
    print("This script will set up crowpeas with uv package manager.\n")
    
    # Check if uv is installed
    if not check_uv_installed():
        return
    
    # Check for CUDA
    cuda_available = detect_cuda()
    
    # Create virtual environment
    venv_path = setup_venv()
    
    # Install PyTorch
    install_pytorch(venv_path, cuda_available)
    
    # Install crowpeas
    install_crowpeas(venv_path)
    
    # Print activation instructions
    print_activation_instructions(venv_path)


if __name__ == "__main__":
    main()