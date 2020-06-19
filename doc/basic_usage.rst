********************
Basic Usage
********************

In the following we are going to discuss the different options and caveats that come with mixing MPI
and pytorch's automatic differentiation (AD) functionality, and what consequences this has for using
the torchmpi library.

Note that although we will within this document mostly talk about the interplay of torchmpi with pytorch's AD,
this does not mean that torchmpi could not in principle be used as one might expect coming from other
MPI libraries. The main difference,
however, is that if one plans to use torchmpi as a building block in some automatic differentiable code,
the usage of torchmpi actually differs a lot to these "classical" programming paradigms.
It is thus *highly* recommended
for everybody to read this document before, e.g., literally translating MPI calls to torchmpi. 

How pytorch's AD works
======================

Since it is important for what follows, we start with a quick reminder on how the AD engine in
pytorch is used. Consider the following code

.. code-block:: python

   import torch
   
   a = torch.tensor([0.5]).requires_grad_()
   b = torch.exp(a)
   b.backward()
   assert(a.grad == b)

This code simply computes the derivative of the function :math:`f(x) = e^x` at the point :math:`x=0.5`.
In the code we do so by initializing a torch tensor ``a`` that has the flag ``requires_grad = True`` set,
which we do here by calling the ``requires_grad_()`` method. This flag is in some sense contagious: Allmost
all torch functions that are called with ``a`` as their argument, pass this flag also to their output. In
the example above this is the exponential function, which returns a tensor ``b`` that has also this flag set.
In addition to this flag ``b`` comes with the info that it was computed from ``a`` and
a property, which is called the gradient function ``grad_fn``.
This is the function that tells pytorch what to do in the backward automatic differentiation pass.

To illustrate this a bit more, consider the following directed acyclic graph (DAG) that represents the
computational flow in the forward phase:

.. graphviz::
   :caption: Forward DAG
   :align: center

   digraph forwarddag {
      exp [shape=rectangle];
      "a" -> exp -> "b";
   }

What now happens in the backward pass is that pytorch executes a reversed DAG of the foward DAG, just with
the functions replaced by their gradient functions. E.g. in the example above this would look like

.. graphviz::
   :caption: Backward DAG
   :align: center

   digraph backwarddag {
      rankdir=BT;
      ExpBackward [shape=rectangle];
      "1" -> ExpBackward -> "a.grad";
   }

In particular, pytorch starts off with :math:`1`, which is obviously the derivative of ``b`` with respect to
itself. It then executes the ``grad_fn`` function, which in this example is the ``ExpBackward`` function.
Not shown in the illustration is that the ``grad_fn`` function internally has a reference to the result of the
forward calculation, which is then  muliplied with ``1`` and defines the output of ``ExpBackward``.
Finally pytorch stores this result in ``a.grad``, which now contains the derviative of ``b`` with respect to ``a``.

This principle of course can be generalized to more complicated DAGs. pytorch in these situations still builds
up the backwards DAG by recording the gradient functions and the dependencies on the go, and then executes this
graph when the ``backward`` method is called. However, there are still some important implications for the
usage in the following, which we want to highlight:

.. _section_pure_functions:

Automatic differentiable functions should at best be pure functions
-------------------------------------------------------------------

This statement is --- if written out like that --- probably not any news,
since the concept of a differentiable function
is a mathematical one, and all mathematical functions are pure in a procedural sense. Hence, a programmatic
representation of a mathematically differentiable function should at best also be pure.

This has some implications. One of the more important ones is, as obvious as it may seem, that this function
needs to have an input and an output. Without an input (and without explicitly modifying the autograd meta data)
the output of a function is from the perspective of the AD engine a constant. The same applies for functions
with no output, whose branch in the backward DAG execution is simply omitted by the AD engine.

Since this is so important, we repeat it:

.. warning::

   **All automatic differentiable functions need to depend on some input tensor,
   and need to return an output tensor**.

DAG edges can only be pytorch tensors of floating point type
------------------------------------------------------------

This goes into the same direction as the last remark. Obviously differentiability is from its mathematical
definition strongly tied to of the real numbers, and the floating point numbers are the only approximation
to them we have in pytorch.

As such we can only exchange floating point tensors along the edges in the DAG.

That some form of additivity is required for the structures that are transported along the DAG edges
can also be seen from the following example

.. code-block:: python

   a = ...
   tmp1 = F(a)
   tmp2 = G1(tmp1)
   tmp3 = G2(tmp2)
   b = H(tmp1, tmp2)
   b.backward()

Note in particular that the output from the node ``F`` is used twice: once as the input for ``G1``
and once as the input for ``G2``. The forward DAG would then look like  

.. graphviz::
   :caption: Forward DAG with bifurcation
   :align: center

   digraph foo {
      F [shape=rectangle];
      G1 [shape=rectangle];
      G2 [shape=rectangle];
      H [shape=rectangle];
      "a" -> F -> G1 -> H -> "b";
      F -> G2 -> H;
   }

The corresponding backward DAG would by simply inverting the arrows and substituting
the function calls by the respective backward function calls, have the form

.. graphviz::
   :caption: Backward DAG with bifurcation
   :align: center

   digraph foo2 {
      rankdir=BT;
      FBackward [shape=rectangle];
      G1Backward [shape=rectangle];
      G2Backward [shape=rectangle];
      HBackward [shape=rectangle];
      "1" -> HBackward -> G1Backward -> FBackward -> "a.grad";
      HBackward -> G2Backward -> FBackward;
   }

However, what this picture does not show is that the bifurctation in the forward evaluation of ``b``
becomes an addition in the backward pass. A more detailed representation of the backward DAG
would thus be

.. graphviz::
   :caption: Backward DAG with explicit addition
   :align: center

   digraph foo2 {
      rankdir=BT;
      FBackward [shape=rectangle];
      "+" [shape=rectangle];
      G1Backward [shape=rectangle];
      G2Backward [shape=rectangle];
      HBackward [shape=rectangle];
      "1" -> HBackward -> G1Backward -> "+" -> FBackward -> "a.grad";
      HBackward -> G2Backward -> "+";
   }

To sum up:

.. warning::

   The edges in the DAG representation can only be pytorch tensors of floating point type.


.. _section_implications_torchmpi:

Implications for torchmpi
=========================

torchmpi is a MPI wrapper library for pytorch tensors that tries to be as *transparent* as possible
to pytorch's AD engine. By transparent we in particular mean that we do not touch the AD engine, but
rather provide the MPI functions as nodes in the DAG that pytorch composes. To be more precise, one should
say the DAGs that pytorch composes, which brings us already to one of the ramifications of this design
decision: When parallelizing your program with torchmpi it is still the case that each MPI rank has its
individual DAG that is run during the backward step. Most importantly, these DAGs do not know anything
about each other, and thus cannot resolve any dependencies with ``requires_grad`` set from any other rank.
As a consequence **it is the sole responsibility of the user to manage these dependencies**.

We will come to it in a minute how the user actually can encode these dependencies, but first start
with an example. Consider the following code, which shows the often used Isend-Recv-Wait idiom.
It from a communication perspective
simply receives a tensor from the left process and passes its own tensor to the right, if all
ranks are imagined to be arranged in a circle.

.. code-block:: python

   import torch
   import torchmpi

   comm = torchmpi.COMM_WORLD

   a = torch.tensor([1.0 + comm.rank]).requires_grad_()

   handle = comm.Isend(a,(comm.rank+1)%comm.size, 0)
   b = comm.Recv(torch.empty_like(a), (comm.rank-1+comm.size)%comm.size, 0)
   comm.Wait(handle)

   res = a+b
   print(res)

This code follows usual MPI coding paradigms and works as expected. However, when we would start asking
for the gradient of (the sum of all) ``res`` with respect to the individual ``a`` s, we would get an incorrect
result.

.. code-block:: python

   res.backward()
   print(a.grad) # <- this would print tensor([1.])

The print function would actually display 1 as the result, whereas taking the derivative of
the sum of all ``res`` variables on all ranks with respect to that specific ``a`` variable should be 2.

This is just one of the things that could happen. There are many more situations, in which the program would
run flawlessly in forward mode, but would e.g. deadlock in the backward pass. To exemplify
how this happens we will look once more at a graphical representation of the DAG.

.. graphviz::
   :caption: Forward DAG for the Isend-Recv-Wait idiom
   :align: center
   :name: naiveforwardisendrecvwaitgraph

   digraph foo2 {
      rank = same;
      subgraph clusterrankm1 {
         a1 [label="a"];
         res1 [label="res"];
         node [shape=rectangle];
         Isend1 [label="Isend"];
         Wait1 [label="Wait"];
         Recv1 [label="Recv"];
         p1 [label="+"];
         a1 -> Isend1 -> Wait1;
         Recv1 -> p1 -> res1;
         a1 -> p1;
         label = "rank - 1";
         color = black;
      };
      subgraph clusterrank {
         a2 [label="a"];
         res2 [label="res"];
         node [shape=rectangle];
         Isend2 [label="Isend"];
         Wait2 [label="Wait"];
         Recv2 [label="Recv"];
         p2 [label="+"];
         a2 -> Isend2 -> Wait2;
         Recv2 -> p2 -> res2;
         a2 -> p2;
         label = "rank";
      }
      subgraph clusterrankp1 {
         a3 [label="a"];
         res3 [label="res"];
         node [shape=rectangle];
         Isend3 [label="Isend"];
         Wait3 [label="Wait"];
         Recv3 [label="Recv"];
         p3 [label="+"];
         a3 -> Isend3 -> Wait3;
         Recv3 -> p3 -> res3;
         a3 -> p3;
         label = "rank + 1";
      }

      Isend1 -> Recv2 [style=dotted, constraint=false];
      Isend2 -> Recv3 [style=dotted, constraint=false];
      #Isend3 -> Recv1 [style=dotted, constraint=false];
   }

The graph as shown above shows the dependencies between the different computations as seen from pytorch's
perspective with the addition of some dotted arrows that show the actual communication that is happening.

If we would now invert the arrows in order to get the corresponding backward DAG we would obtain

.. graphviz::
   :caption: Backward DAG for the Isend-Recv-Wait idiom for a single rank
   :align: center

   digraph foo2 {
      rankdir=BT;
      subgraph clusterrankm1 {
         a1 [label="a.grad"];
         res1 [label="1"];
         node [shape=rectangle];
         Isend1 [label="IsendBackward", style=filled, color=gray];
         Wait1 [label="WaitBackward", style=filled, color=gray];
         Recv1 [label="RecvBackward", style=filled, color=gray];
         p1 [label="AddBackward0"];
         Wait1 -> Isend1 -> a1;
         res1 -> p1 -> Recv1;
         p1 -> a1;
         label = "rank";
      };
   }

This graph immanently makes clear why ``a.grad`` contains 1 in the end. All grayed-out nodes are omitted
--- or to be more precise, not even generated --- by pytorch's AD engine, such that only ``AddBackward0``
is called, which just passes through 1 to ``a.grad``.

From this discussion and the :ref:`naiveforwardisendrecvwaitgraph` it becomes apparent that there are some parts
that are implicit in the program code but that are missing in the DAG representation:

#. As noted earlier, the DAGs are local to each MPI rank, and they do not resolve any dependencies that
   are the effect of communication.
#. The DAGs also lack any information that was present in the linear ordering of commands in the source code
   file. E.g. the ``Recv`` call has to happen after ``Isend``, and ``Wait`` has to happen after ``Recv``.

**It is the users responsibility to encode these dependencies in the DAG!.**
This brings us to the tools torchmpi provides to mitigate this situation.

The first one is a direct consequence of the discussion in the section on
:ref:`pure functions <section_pure_functions>`: all DAG nodes need an input and an output.
In our example above, this would e.g. concern the :py:meth:`torchmpi.MPI_Communicator.Wait`
call. In principle, ``MPI_Wait`` does not return a floating point tensor. However, torchmpi
returns a floating-point tensor, giving the user the possibility to use it to encode
any other dependencies on the ``Wait`` call. These tensors are named **dummies** in torchmpi.
They do not convey any other information than that there is some (virtual/artificial)
dependency to be encoded in the DAG.

The dummies themselves are not really useful without a way to join them with the DAG. This is
what the :py:func:`torchmpi.JoinDummies` function is actually for. The call signature of
:py:func:`torchmpi.JoinDummies` is given by

.. code-block:: python

   def JoinDummies(loopthrough: torch.Tensor, dummies: List[torch.Tensor]) -> torch.Tensor

The function takes two arguments: the loopthrough variable and a list of dummies. From a forward
execution perspective the ``JoinDummies`` function is a no-op, it simply --- as the name suggests ---
loops through the ``loopthrough`` variable. The ``dummies`` are discarded and not used.

However, pytorch does not know about this behaviour of the ``JoinDummies`` function, and considers
the result of the function to actually depend on the dummies. Consequently, pytorch will also
respect this dependency in the backward DAG.

The :py:func:`torchmpi.JoinDummies` function also has a sister function :py:func:`torchmpi.JoinDummiesHandle`, which
is thought for situations in which the ``loopthrough`` variable is a :py:class:`torchmpi.WaitHandle`
from a non-blocking MPI call, as e.g. returned by :py:func:`torchmpi.MPI_Communicator.Isend`. The signature
of :py:func:`torchmpi.JoinDummiesHandle` is

.. code-block:: python

   def JoinDummiesHandle(handle: WaitHandle, dummies: List[torch.Tensor]) -> WaitHandle

Returning to the Isend-Recv-Wait example, we now want to put these tools to use. Starting with
the call to :py:func:`torchmpi.MPI_Communicator.Recv`, we want this call to happen after
:py:func:`torchmpi.MPI_Communicator.Isend`. Note that ``Isend`` returns a ``WaitHandle``, which
cannot directly be passed to ``JoinDummies``. For these situations we will use the 
:py:attr:`torchmpi.WaitHandle.dummy` property, which gives us a means to convert a ``WaitHandle``
to a dummy tensor. In the example from above this could then
look like

.. code-block:: python

   handle = comm.Isend(a,(comm.rank+1)%comm.size, 0)
   recvbuffer = torchmpi.JoinDummies(torch.empty_like(a), [handle.dummy])
   #                                 ~~~~~~~~~~~~~~~~~~~
   #                                 This is what we
   #                                 originally wanted
   #                                 to pass to Recv
   #                                                      ~~~~~~~~~~~~~~
   #                                                      This adds the handle
   #                                                      from the previous Isend call
   #                                                      as a dummy dependency to the DAG
   b = comm.Recv(recvbuffer, (comm.rank-1+comm.size)%comm.size, 0)

For the ``Wait`` we now also want this to happen after the ``Recv`` call. This time we make use of
:py:func:`torchmpi.JoinDummiesHandle`.

.. code-block:: python
 
   b = comm.Recv(recvbuffer, (comm.rank-1+comm.size)%comm.size, 0)
   wait_ret = comm.Wait(torchmpi.JoinDummiesHandle(handle,[b]))

Note that we already added a return variable for ``Wait``, since we still want to encode
that our end result, the (implicit) sum of all ``res`` on all ranks, depends on the ``Isend`` to
have finished. For that we introduce another call to :py:func:`torchmpi.JoinDummies`.

.. code-block:: python

   wait_ret = comm.Wait(torchmpi.JoinDummiesHandle(handle,[b]))

   res = torchmpi.JoinDummies(a+b, [wait_ret])


The full code example now looks like

.. code-block:: python

   import torch
   import torchmpi

   comm = torchmpi.COMM_WORLD

   a = torch.tensor([1.0 + comm.rank]).requires_grad_()

   handle = comm.Isend(a,(comm.rank+1)%comm.size, 0)
   recvbuffer = torchmpi.JoinDummies(torch.empty_like(a), [handle.dummy])
   b = comm.Recv(recvbuffer, (comm.rank-1+comm.size)%comm.size, 0)
   wait_ret = comm.Wait(torchmpi.JoinDummiesHandle(handle,[b]))

   res = torchmpi.JoinDummies(a+b, [wait_ret])
   print(res)

   res.backward()
   print(a.grad) # <- this would now correctly print tensor([2.])

This code would now print the correct result for ``a.grad``. To exemplify the differences to the
first version of the code we will also look at the DAG of the new version

.. graphviz::
   :caption: Forward DAG for the Isend-Recv-Wait idiom with dummy dependencies
   :align: center

   digraph foo2 {
      subgraph clusterrankm1 {
         a [label="a"];
         res [label="res"];
         node [shape=rectangle];
         JoinDummies1 [label="JoinDummies"];
         JoinDummiesHandle [label="JoinDummiesHandle"];
         JoinDummies2 [label="JoinDummies"];
         Isend [label="Isend"];
         Wait [label="Wait"];
         Recv [label="Recv"];
         p1 [label="+"];
         a -> Isend;
         Isend -> JoinDummies1 -> Recv;
         Recv -> JoinDummiesHandle -> Wait;
         Isend -> JoinDummiesHandle;
         Recv -> p1 -> JoinDummies2;
         a -> p1;
         Wait -> JoinDummies2;
         JoinDummies2 -> res;
         label = "rank";
         color = black;
      };
   }

The important point to note is that all communciation is part of a path between
``a`` and ``res``, and in comparison to the first version of the code there are no "dead branches".
pytorch's AD engine thus has to call the respective backward methods when it propagates the gradient
back from ``res`` to ``a.grad``.

.. warning::

   In general, if you write a function that uses torchmpi internally and shall be automatic differentiable,
   make sure that all communication primitives are through one way or another part of a DAG path
   that connects input and output of that function.


