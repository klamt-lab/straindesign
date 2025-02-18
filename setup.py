from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop  # Added import for develop
import os, subprocess, sys


class CustomInstallCommand(install):

    def run(self):
        print("Welcome to the straindesign installation process!")
        # Ensure JAVA_HOME is set by searching common locations if needed
        if self.search_for_jvm():
            install.run(self)  # Proceed with normal installation
            return

        print("Java not found. Attempting to install OpenJDK...")
        self.install_openjdk()
        install.run(self)  # Proceed after installation

    def search_for_jvm(self):
        common_java_paths = [
            "C:\\Program Files\\Java",  # Windows
            "/usr/lib/jvm",  # Linux
            "/Library/Java/JavaVirtualMachines",  # macOS
            os.path.dirname(sys.executable)
        ]
        for base in common_java_paths:
            if os.path.exists(base):
                for root, _dirs, files in os.walk(base):
                    if any(lib in files for lib in ["jvm.dll", "libjvm.so", "libjvm.dylib"]):
                        return root
        return None

    def install_openjdk(self):
        # Check if conda is available
        try:
            subprocess.check_call(["conda", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            conda_available = True
        except Exception:
            conda_available = False

        if conda_available:
            print("Conda is available. Installing openjdk via conda-forge...")
            try:
                subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", "openjdk"])
                return
            except Exception:
                print("Conda installation of openjdk failed, falling back to pip-based installation.")

        # Fallback: pip-based installation using install-jdk
        try:
            import jdk
        except ImportError:
            print("install-jdk not found. Installing it now via pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "install-jdk"])
            import jdk
        try:
            base_path = os.path.dirname(sys.executable)
            install_path = os.path.join(base_path, "openjdk")
            jdk.install('17', path=install_path, jre=True)
            print(f"OpenJDK successfully installed at {install_path}")
        except Exception as e:
            print("Automatic installation of OpenJDK failed. Please install OpenJDK manually.")
            sys.exit(1)


class CustomDevelopCommand(develop):

    def run(self):
        print("Running custom develop command...")
        # Re-use the OpenJDK installation logic from our CustomInstallCommand
        installer = CustomInstallCommand(self.distribution)
        if not installer.search_for_jvm():
            print("Java not found. Attempting to install OpenJDK before development installation...")
            installer.install_openjdk()
        develop.run(self)


setup(
    name="straindesign",
    version="1.14",
    url="https://github.com/klamt-lab/straindesign.git",
    description="Computational strain design package for the COBRApy framework",
    long_description=("Computational strain design package for the COBRApy framework, offering standard and advanced "
                      "tools for the analysis and redesign of biological networks"),
    long_description_content_type="text/plain",
    author="Philipp Schneider",
    author_email="zgddtgt@gmail.com",
    license="Apache License 2.0",
    python_requires=">=3.7",
    package_data={"straindesign": ["efmtool.jar"]},
    packages=find_packages(),
    install_requires=["cobra", "jpype1", "scipy", "matplotlib", "psutil"],
    project_urls={
        "Bug Reports": "https://github.com/klamt-lab/straindesign/issues",
        "Source": "https://github.com/klamt-lab/straindesign/",
        "Documentation": "https://straindesign.readthedocs.io/en/latest/index.html"
    },
    classifiers=[
        "Intended Audience :: Science/Research", "Development Status :: 3 - Alpha", "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", "Programming Language :: Python :: 3.11", "Programming Language :: Python :: 3.12",
        "Natural Language :: English", "Operating System :: OS Independent", "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    keywords=["metabolism", "constraint-based", "mixed-integer", "strain design"],
    zip_safe=False,
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand
    },
)
