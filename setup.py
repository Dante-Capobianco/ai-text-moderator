from setuptools import setup, find_packages

setup(
    name="your_package_name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here, e.g.
        # "tensorflow-gpu>=2.4.0",
        # add scikit-;earn, numpy
    ],
    python_requires='>=3.7, <4',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)

