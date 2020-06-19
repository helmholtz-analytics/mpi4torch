************
torchmpi
************

torchmpi is an automatic-differentiable wrapper of MPI functions for the pytorch tensor library.

MPI stands for Message Passing Interface and is the de facto standard communication interface on
high-performance computing resources. To facilitate the usage of pytorch on these resources an MPI wrapper
that is transparent to pytorch's automatic differentiation (AD) engine is much in need. This library tries
to bridge this gap.

Note that this library is quite lowlevel and tries to stick as closely as possible to the MPI function
names. If you need more convenience, please consult the
`HeAT library <https://github.com/helmholtz-analytics/heat>`_
which (hopefully soon) will internally use this library.


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
