"""
Setup script for Customer Segmentation Package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Customer Segmentation for E-commerce"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name="customer-segmentation",
    version="1.0.0",
    author="Customer Segmentation Team",
    author_email="team@shoppy.com",
    description="Customer segmentation pipeline for e-commerce businesses",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/shoppy/customer-segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Point-Of-Sale",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
        ],
        "advanced": [
            "hdbscan>=0.8.29",
            "numba>=0.57.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "customer-segmentation=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "customer-segmentation",
        "rfm-analysis",
        "clustering",
        "kmeans",
        "e-commerce",
        "machine-learning",
        "data-science",
    ],
    project_urls={
        "Bug Reports": "https://github.com/shoppy/customer-segmentation/issues",
        "Source": "https://github.com/shoppy/customer-segmentation",
        "Documentation": "https://customer-segmentation.readthedocs.io/",
    },
)
