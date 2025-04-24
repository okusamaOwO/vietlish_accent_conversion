from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='monotonic_align',
    packages=["monotonic_align"],
    ext_modules=cythonize("monotonic_align/core.pyx"),
    include_dirs=[numpy.get_include()]
)
