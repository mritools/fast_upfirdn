#!/usr/bin/env python
import os
import sys
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

PACKAGES = find_packages()

# Get version and release info, which is all stored in fast_upfirdn/version.py
ver_file = os.path.join("fast_upfirdn", "version.py")
with open(ver_file) as f:
    exec(f.read())
# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
SETUP_REQUIRES = ["setuptools >= 24.2.0"]
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ["wheel"] if "bdist_wheel" in sys.argv else []

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

    # check that cython version is > 0.23
    cython_version = cython.__version__
    if float(cython_version.partition(".")[2][:2]) < 21:
        raise ImportError
    build_cython = True
except ImportError:
    build_cython = False
    # TODO: allow a python-only installation with reduced functionality
    raise EnvironmentError(
        """
        cython could not be found.  Compilation of fast_upfirdn requires Cython
        version >= 0.23.
        Install or upgrade cython via:
        pip install cython --upgrade
        """
    )

extra_compile_args = []
cmdclass = {"build_ext": build_ext}

src_path = os.path.join("fast_upfirdn", "cpu")

# C extensions
ext_upfirdn = Extension(
    "fast_upfirdn.cpu._upfirdn_apply",
    sources=[os.path.join(src_path, "_upfirdn_apply.pyx")],
    language="c",
    extra_compile_args=extra_compile_args,
    include_dirs=[numpy_include],
)

ext_modules = [ext_upfirdn]

c_macros = [("PY_EXTENSION", None)]
cython_macros = []
cythonize_opts = {"language_level": "3"}
if os.environ.get("CYTHON_TRACE"):
    cythonize_opts["linetrace"] = True
    cython_macros.append(("CYTHON_TRACE_NOGIL", 1))

if USE_CYTHON:
    ext_modules = cythonize(ext_modules, compiler_directives=cythonize_opts)

opts = dict(
    name=NAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    install_requires=REQUIRES,
    python_requires=PYTHON_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    requires=REQUIRES,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)


if __name__ == "__main__":
    setup(**opts)
