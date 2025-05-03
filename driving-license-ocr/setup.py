from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
setup(
    name="sl_license_ocr",
    version="1.0",
    author="Randika Prabashwara",
    author_email="randikap.20@cse.mrt.ac.lk",
    description="Machine learning pipeline for Sri Lankan driving license information extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/randikapra/driving-license-ocr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sl-license-ocr=src.cli:main",
        ],
    },
    include_package_data=True,
)