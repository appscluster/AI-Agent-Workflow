#!/usr/bin/env python
"""
Installation script for AI Agent Workflow
"""
import os
import sys
import subprocess
import argparse

def check_python_version():
    """Check that Python version is 3.10 or higher"""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        print(f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}")
        return False
    return True

def install_requirements(install_optional=False):
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        # Install base requirements - continue even if some packages fail
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Some packages failed to install: {e}")
            print("Trying to install core dependencies individually...")
            
            # Install core dependencies individually
            core_packages = [
                "llama-index>=0.9.0",
                "llama-index-agent-openai>=0.1.0", 
                "llama-index-embeddings-openai>=0.1.0",
                "llama-index-llms-openai>=0.1.0",
                "llama-index-readers-file>=0.1.0",
                "llama-index-vector-stores-chroma>=0.1.0",
                "PyMuPDF>=1.22.0",
                "pydantic>=2.0.0",
                "faiss-cpu>=1.7.4",
                "numpy>=1.24.0",
                "python-dotenv>=1.0.0"
            ]
            
            for package in core_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError:
                    print(f"Warning: Failed to install {package}")
        
        # Install optional packages if requested
        if install_optional:
            print("Installing optional packages for sample document generation...")
            optional_packages = [
                "faker", 
                "python-docx", 
                "matplotlib", 
                "numpy",
                "docx2pdf"  # For PDF conversion
            ]
            
            for package in optional_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError:
                    print(f"Warning: Failed to install optional package {package}")
        
        print("Package installation completed with some dependencies!")
        print("Note: Some advanced features may not be available due to missing packages.")
        return True
    
    except Exception as e:
        print(f"Error during installation: {e}")
        return False

def setup_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = ".env"
    template_file = ".env.example"
    
    if os.path.exists(env_file):
        print(f"{env_file} already exists. Skipping creation.")
        return
    
    if not os.path.exists(template_file):
        print(f"Warning: {template_file} not found. Please create .env file manually.")
        return
    
    # Copy the template
    with open(template_file, 'r') as template, open(env_file, 'w') as env:
        env.write(template.read())
    
    print(f"Created {env_file} from template. Please edit it to add your API keys.")

def create_sample_dirs():
    """Create necessary directories"""
    dirs = ["samples", "output"]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

def main():
    """Main installation function"""
    parser = argparse.ArgumentParser(description="Install AI Agent Workflow")
    parser.add_argument('--with-samples', action='store_true', help='Install packages for sample document generation')
    args = parser.parse_args()
    
    print("=== AI Agent Workflow Installation ===")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements(args.with_samples):
        return
    
    # Setup environment file
    setup_env_file()
    
    # Create sample directories
    create_sample_dirs()
    
    print("\nInstallation completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file to add your API keys")
    print("2. Generate a sample document or use your own PDF")
    print("3. Run the system with: python main.py --document path/to/your/document.pdf")
    
    if args.with_samples:
        print("\nTo generate a sample document:")
        print("python utils/sample_doc_generator.py")

if __name__ == "__main__":
    main()
