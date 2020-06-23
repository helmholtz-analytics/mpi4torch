
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import copy
import os

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
    versiontext = filehandle.read()

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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
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
        'torch>=1.5.0',
        'mpi4py'
    ]
)
