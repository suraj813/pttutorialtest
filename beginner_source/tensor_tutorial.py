"""
Tensors
--------------------------------------------

Tensors are a specialized data structure that are very similar to arrays
and matrices. In PyTorch, we use tensors to encode the inputs and
outputs of a model, as well as the model’s parameters.

Tensors are similar to NumPy’s ndarrays, except that tensors can run on
GPUs or other specialized hardware to accelerate computing. If you’re familiar with ndarrays, you’ll
be right at home with the Tensor API. If not, follow along in this quick
API walkthrough.

"""

import torch
import numpy as np


######################################################################
# Tensor Initialization
# ~~~~~~~~~~~~~~~~~~~~~
#
# Tensors can be initialized in various ways. Take a look at the following examples
#
# **Directly from data**
# Tensors can be created directly from data. The data type is automatically inferred.

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

######################################################################
# **From a NumPy array**
# Tensors can be created from NumPy arrays (and vice versa - see :ref:`bridge-to-np-label`).
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


###############################################################
# **From another tensor:**
# The new tensor retains the properties (shape, datatype) of the arg tensor, unless explicitly overridden.

x_ones = torch.ones_like(x_data) # 2 x 2 matrix of ones

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)


######################################################################
# **With random or constant values:**
#
# ``shape`` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor:\n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}")




######################################################################
# --------------
#


######################################################################
# Tensor Attributes
# ~~~~~~~~~~~~~~~~~
#
# Tensor attributes describe their shape, datatype, and the device on which they are stored.

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor lives on: {tensor.device}")


######################################################################
# --------------
#


######################################################################
# Tensor Operations
# ~~~~~~~~~~~~~~~~~
#
# Over 100 tensor operations, including transposing, indexing, slicing,
# mathematical operations, linear algebra, random sampling, and more are
# comprehensively described
# `here <https://pytorch.org/docs/stable/torch.html>`__.
#
# Each of them can be run on the GPU (at typically higher speeds than on a
# CPU). If you’re using Colab, allocate a GPU by going to Edit > Notebook
# Settings.
#

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')


######################################################################
# Try out some of the operations from the list.
# If you're familiar with the NumPy API, you'll find the Tensor API a breeze to use.
#

###############################################################
# **Standard numpy-like indexing and slicing:**

tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

######################################################################
# **Joining tensors** You can use ``torch.cat`` to concatenate a sequence of tensors along a given dimension.
# See also `torch.stack <https://pytorch.org/docs/stable/generated/torch.stack.html>`__,
# another tensor joining op that is subtly different from ``torch.cat``.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

######################################################################
# **Multiplying tensors**

# This computes the element-wise product
print(tensor.mul(tensor))
# Alternative syntax:
print(tensor * tensor)

# This computes the matrix multiplication between two tensors
print(tensor.matmul(tensor.T))
# Alternative syntax:
print(tensor @ tensor.T)


######################################################################
# **In-place operations**
# Operations that have a '_' suffix are in-place. For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.

print(tensor)
tensor.add_(5)
print(tensor)

######################################################################
# .. note::
#      In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss
#      of history. Hence, their use is discouraged.

######################################################################
# --------------
#


######################################################################
# .. _bridge-to-np-label:
#
# Bridge with NumPy
# ~~~~~~~~~~~~~~~~~
# Tensors on the CPU and NumPy arrays can share their underlying memory
# locations, and changing one will change	the other.


######################################################################
# Tensor to NumPy array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
a = torch.ones(5)
print(f"a: {a}")
b = a.numpy()
print(f"b: {b}")

######################################################################
# A change in ``a`` reflects in ``b``

a.add_(1)
print(f"a: {a}")
print(f"b: {b}")


######################################################################
# NumPy array to Tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(f"a: {a}")
print(f"b: {b}")
