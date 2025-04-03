from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from requirements.txt."""
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def read_readme():
    """Read README.md for long description."""
    with open('README.md', encoding='utf-8') as f:
        return f.read()

setup(
    name="neuromind",
    version="0.1.0",
    description="A powerful framework for building AI agents with advanced memory capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/neuromind-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "neuromind=neuromind.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/neuromind-framework/issues",
        "Source": "https://github.com/yourusername/neuromind-framework",
        "Documentation": "https://github.com/yourusername/neuromind-framework/docs",
    },
) 