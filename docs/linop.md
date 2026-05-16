# `linop` — Base classes and algebraic operators

## Base classes

```{eval-rst}
.. autoclass:: codeop.linop.LinOp
   :members:
   :show-inheritance:

.. autoclass:: codeop.linop.BaseOp
   :members:
   :show-inheritance:
```

## Algebraic operators

```{eval-rst}
.. autoclass:: codeop.linop.Adjoint
   :members:
   :show-inheritance:

.. autoclass:: codeop.linop.Scaled
   :members:
   :show-inheritance:

.. autoclass:: codeop.linop.Symmetric
   :members:
   :show-inheritance:

.. autoclass:: codeop.linop.Explicit
   :members:
   :show-inheritance:
```

## Compound operators

```{eval-rst}
.. autoclass:: codeop.linop.ProdOp
   :members:
   :show-inheritance:

.. autoclass:: codeop.linop.AddOp
   :members:
   :show-inheritance:

.. autoclass:: codeop.linop.SubOp
   :members:
   :show-inheritance:

.. autoclass:: codeop.linop.VStack
   :members:
   :show-inheritance:

.. autoclass:: codeop.linop.HStack
   :members:
   :show-inheritance:
```

## Types and utilities

```{eval-rst}
.. autofunction:: codeop.linop.vectorize

.. autofunction:: codeop.linop.unvectorize

.. autofunction:: codeop.linop.asmatrix
```
