
metagraph-numba Documentation
=============================

metagraph-numba is a plugin for `Metagraph`_ that enables algorithms to be
just-in-time (JIT) compiled with Numba.  This package currently does not
include any graph algorithm implementations in Numba, and is mostly used to
support the `metagraph-mlir`_ plugin.  To learn more about compiler plugins in
Metagraph, see the `Compiler Plugins`_ section of the Metagraph Plugin Author
Guide. 

metagraph-numba is licensed under the `Apache 2.0 license`_ and the source
code can be found on `Github`_.


.. _Metagraph: https://metagraph.readthedocs.org
.. _metagraph-mlir: https://metagraph-mlir.readthedocs.org
.. _Compiler Plugins: https://metagraph.readthedocs.org/en/plugin_author_guide/compiler_plugins.html
.. _Apache 2.0 license: https://www.apache.org/licenses/LICENSE-2.0
.. _Github: https://github.com/metagraph-dev/metagraph-numba


Installation
------------

metagraph-numba is currently only distributed via conda.  To install::

    conda install -c metagraph -c conda-forge metagraph-numba


Implementing Algorithms with metagraph-numba
--------------------------------------------

The metagraph-numba compiler assumes that the decorated algorithm function is
compilable with Numba directly.  For example, suppose an abstract algorithm
has already been defined to add two vectors:

.. code-block:: python

   @abstract_algorithm("example.add")
   def example_add(a: Vector, b: Vector) -> Vector:
       pass

A concrete implementation of this algorithm can be written this way:

.. code-block:: python

   @concrete_algorithm("example.add", compiler="numba")
   def compiled_add(a: NumpyVectorType, b: NumpyVectorType) -> NumpyVectorType:
      return a + b

Numba has support for the input and output types, which are 1D NumPy arrays,
and the body of the function, which does array addition.  For more details on
what Numba supports, see the `Numba Supported Python Features`_ and `Numba
Supported NumPy Features`_.

.. _Numba Supported Python Features: https://numba.readthedocs.io/en/stable/reference/pysupported.html
.. _Numba Supported NumPy Features: https://numba.readthedocs.io/en/stable/reference/numpysupported.html

Internally, the compiler wraps each algorithm function in a
``numba.jit(inline="always")`` call.  This pulls every algorithm body in the
task subgraph into the generated wrapper function, allowing for more
optimization opportunies.  Note that the resulting function is not
JIT-compiled until the first call, which allows for further specialization
on argument attributes like the dtypes of input arrays.
