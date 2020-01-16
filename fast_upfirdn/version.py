from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 2
_version_micro = ""  # use "" for first of series, number for 1 and above
_version_extra = ""
# _version_extra = ""  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = ".".join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]

# Description should be a one-liner:
description = "fast_upfirdn: CuPy-based implementation of scipy.signal.upfirdn"
# Long description will go up on the pypi page
long_description = """

GPU upfirdn
===========
The core low-level function implemented here is an equivalent of
``scipy.signal.upfirdn`` but with support for both CPU (via [NumPy]) and GPU
(via [CuPy]).

The version of ``upfirdn`` here supports several signal extension modes. These
have been contributed upstream to SciPy and are available there for SciPy 1.4+.

.. _README: https://github.com/grlee77/fast_upfirdn/blob/master/README.md

License
=======
``fast_upfirdn`` is licensed under the terms of the BSD 3-clause license. See the
file "LICENSE" for information on the history of this software, terms &
conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2019--,
Gregory R. Lee, Cincinnati Children's Hospital Medical Center.
"""

NAME = "fast_upfirdn"
MAINTAINER = "Gregory R. Lee"
MAINTAINER_EMAIL = "grlee77@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/grlee77/fast_upfirdn"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Gregory R. Lee"
AUTHOR_EMAIL = "grlee77@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {"fast_upfirdn": [pjoin("data", "*"), pjoin("tests", "*")]}
REQUIRES = ["numpy", "cython"]
PYTHON_REQUIRES = ">= 3.6"
