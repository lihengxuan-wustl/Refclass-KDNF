try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize

setup(ext_modules = cythonize(
		"setcoverinc.pyx",
		language="c++",              # this causes Pyrex/Cython to create C++ source
    )
)