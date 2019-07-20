#!/usr/bin/env python
import os
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# cython check
try:
    import cython
    # check that cython version is > 0.21
    cython_version = cython.__version__
    if float(cython_version.partition(".")[2][:2]) < 21:
        raise ImportError
    build_cython = True
except:
    build_cython = False
    # TODO: allow a python-only installation with reduced functionality
    raise EnvironmentError(
        """
        cython could not be found.  Compilation of pyir.upfirdn requires Cython
        version >= 0.21.
        Install or upgrade cython via:
        pip install cython --upgrade
        """)

extra_compile_args = []
cmdclass = {'build_ext': build_ext}

src_path = os.path.join('fast_upfirdn', 'cpu')

# C extensions
ext_upfirdn = Extension(
    'fast_upfirdn.cpu._upfirdn_apply',
    sources=[os.path.join(src_path, '_upfirdn_apply.pyx'), ],
    language='c',
    extra_compile_args=extra_compile_args,
    include_dirs=[numpy_include, ])

ext_modules = [ext_upfirdn, ]

c_macros = [("PY_EXTENSION", None)]
cython_macros = []
cythonize_opts = {}
if os.environ.get("CYTHON_TRACE"):
    cythonize_opts['linetrace'] = True
    cython_macros.append(("CYTHON_TRACE_NOGIL", 1))

if USE_CYTHON:
    ext_modules = cythonize(ext_modules, compiler_directives=cythonize_opts)

setup(
    name="fast_upfirdn",
    packages=find_packages(),
    version="0.1",
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    description="GPU and CPU implementations of upfirdn and convolve.",
    author="Gregory R. Lee",
    author_email="grlee77@gmail.com",
    # url='',
    license="BSD 3-clause",
    zip_safe=False,
)
