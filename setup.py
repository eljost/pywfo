from os import path

# from packaging.version import parse
from setuptools import find_packages, setup
import sys

if sys.version_info.major < 3:
    raise SystemExit("Python 3 is required!")

# Read contents of the README for use in the description on PyPI
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md")) as handle:
    long_description = handle.read()

setup(
    name="pywfo",
    # version=parse("0.0.1"),
    version="0.0.1",
    description="Wavefunction overlaps in python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eljost/pywfo",
    maintainer="Johannes Steinmetzer",
    maintainer_email="johannes.steinmetzer@uni-jena.de",
    license="License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    platforms=["unix"],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
    ],
)
