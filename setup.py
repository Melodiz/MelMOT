"""
Setup script for MelMOT package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="melmot",
    version="2.0.0",
    author="Ivan Novosad",
    author_email="ivan.novosad@example.com",
    description="Multi-Object Tracking for Retail Spaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/melmot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "melmot=melmot.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "melmot": ["config/*.yaml", "config/*.json"],
    },
)
