.. image:: _static/img/mpi4torch-logo-extrawhitespace.svg

mpi4torch is an automatic-differentiable wrapper of MPI functions for the pytorch tensor library.

MPI stands for Message Passing Interface and is the de facto standard communication interface on
high-performance computing resources. To facilitate the usage of pytorch on these resources an MPI wrapper
that is transparent to pytorch's automatic differentiation (AD) engine is much in need. This library tries
to bridge this gap.

.. toctree::
   :maxdepth: 3
   :caption: Table of Contents

   basic_usage
   examples
   api_reference
   glossary

Indices and tables
==================

 * :ref:`genindex`
 * :ref:`modindex`
 * :ref:`search`
