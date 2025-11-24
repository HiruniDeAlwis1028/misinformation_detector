import subprocess
import sys

def install_packages():
    packages = [
        "torch",
        "transformers", 
        "datasets",
        "pandas",
        "numpy",
        "accelerate",
        "psutil",
        "flask",
        "flask_cors",
        "scikit-learn",
        "requests"
    ]
    
    print("🚀 Installing required packages...")
    print("This may take 5-10 minutes...")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Installed: {package}")
        except Exception as e:
            print(f"Failed to install {package}: {e}")
    
    print("\nAll packages installed successfully!")
    print("Next: Run python create_large_dataset.py")

if __name__ == "__main__":
    install_packages()
