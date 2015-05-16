# coding=utf-8
from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules = cythonize([
        "solution/*.pyx"
        # "method_1.pyx",
        # "method_2.pyx",
        # "main_algorithm.pyx"
    ]),
    include_dirs=[numpy.get_include()]
)