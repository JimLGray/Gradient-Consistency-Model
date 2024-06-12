from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    name='Spline MV Disparity',
    package_dir={"": "src/"},
    packages=find_packages(where="src/basic/"),
    ext_modules=cythonize([
        "src/basic/cgrad.pyx",
        "src/basic/graph.pyx",
        'src/splines.pyx',
        'src/basic/simple.pyx',
        'src/basic/warp.pyx',
        'src/basic/map_depth.cpp'
    ], annotate=True, language_level="3"),
    include_dirs=[numpy.get_include()]
)
