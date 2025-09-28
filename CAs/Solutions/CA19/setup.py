"""
Setup script for CA19 Modular package
"""

from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def read_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="ca19-modular",
    version="1.0.0",
    author="CA19 Research Team",
    author_email="research@ca19.team",
    description="Modular implementations of advanced RL systems from CA19",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ca19-research/ca19-modular",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "quantum": ["qiskit>=0.39.0", "qiskit-aer>=0.11.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "qiskit>=0.39.0",
            "qiskit-aer>=0.11.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "ca19_modular": ["*.md", "*.txt"],
    },
    keywords=[
        "reinforcement-learning",
        "quantum-computing",
        "neuromorphic",
        "artificial-intelligence",
        "machine-learning",
        "deep-learning",
        "quantum-ml",
        "spiking-neural-networks",
    ],
    project_urls={
        "Bug Reports": "https://github.com/ca19-research/ca19-modular/issues",
        "Source": "https://github.com/ca19-research/ca19-modular",
        "Documentation": "https://ca19-modular.readthedocs.io/",
    },
    zip_safe=False,
)
