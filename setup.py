
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import torch
import copy
import os
import sys

try:
    # importlib only got added in cpython 3.8
    if sys.version_info >= (3, 8):
        from importlib import metadata as importlib_metadata
    else:
        # cf. pyproject.toml file, which makes pip install importlib-metadata if necessary
        import importlib_metadata

    torchversion = importlib_metadata.distribution("torch").version
except:
    # Fallback, but this should never happen.
    torchversion = torch.__version__.split("+")[0]

class MpiBuildExtension(BuildExtension):
    def __init__(self, *args,**kwargs):
        super(MpiBuildExtension,self).__init__(*args,use_ninja=False,**kwargs)

    def build_extensions(self):
        """
            This code makes a lot assumptions on distutils internal implementation of
            UnixCCompiler class. However, it seems to be standard to make these assumptions,
            as PyTorch and mpi4py also make these assumptions.

            TODO: Obviously this only works for unix systems
        """

        # Save original compiler and reset it later on
        original_compiler = self.compiler.compiler_so
        new_compiler = copy.deepcopy(original_compiler)
        new_compiler[0] = 'mpicc'
        # Save original CXX compiler and reset it later on

        # distutils' UnixCCompiler likes to use the C++ compiler for linking, so we set it manually
        original_cxx_compiler = self.compiler.compiler_cxx
        new_cxx_compiler = copy.deepcopy(original_cxx_compiler)
        new_cxx_compiler[0] = 'mpicxx'
        # Save original linker and reset it later on
        # should not be used, but we set it anyway
        original_linker = self.compiler.linker_so
        new_linker = copy.deepcopy(original_linker)
        new_linker[0] = 'mpicc'
        try:
            self.compiler.set_executable('compiler_so', new_compiler)
            self.compiler.set_executable('compiler_cxx', new_cxx_compiler)
            self.compiler.set_executable('linker_so', new_linker)
            BuildExtension.build_extensions(self)
        finally:
            self.compiler.set_executable('compiler_so', original_compiler)
            self.compiler.set_executable('compiler_cxx', original_cxx_compiler)
            self.compiler.set_executable('linker_so', original_linker)

with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as filehandle:
    long_description = filehandle.read()

with open(os.path.join(os.path.dirname(__file__), 'version.txt'), encoding='utf-8') as filehandle:
    versiontext = filehandle.read().rstrip()

setup(
    name='mpi4torch',
    version=versiontext,
    description='AD-compatible implementation of several MPI functions for pytorch tensors',
    author='Philipp Knechtges',
    author_email='philipp.knechtges@dlr.de',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    package_dir = {'mpi4torch': 'src'},
    packages = ['mpi4torch'],
    ext_modules=[
        CppExtension(
            name='mpi4torch._mpi',
            sources=['csrc/extension.cpp'],
            extra_compile_args=['-g']),
    ],
    cmdclass={
        'build_ext': MpiBuildExtension
    },
    install_requires=[
        # Pin the required pytorch version of the final binary wheels
        # to the pytorch version used at build-time. This way we
        # avoid possible ABI-incompatibilities.
        'torch==' + torchversion,
    ]
)
