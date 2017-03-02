# TensorFlow for Vala

This package contains experimental Vala bindings for TensorFlow.

## Current Status ##

Highly experimental [Vala][vala] [bindings][vlb] for [TensorFlow][tensorflow],
using the [C API][c_api].

As a result, current support for TensorFlow is limited by the exposed functionality in the API,
which can be found in the [official site][tf_bindings].

| Feature                  | Python       | C API       | Vala     |
| ------------------------ | ------------ | ----------- | -------- |
| Run a predefined Graph   | Yes          | Yes         | Yes      |
| Graph construction       | Yes          | Yes         | Yes      |
| Gradients                | Yes          | No          | No       |
| Functions                | Yes          | No          | No       |
| Control Flow             | Yes          | No          | No       |
| Neural Network Library   | Yes          | No          | No       |

## How to build ##

### Install the TensorFlow C library

This step is the same as the one for the [Rust bindings][rust_bindings]:

1. Install [SWIG](http://www.swig.org) and [NumPy](http://www.numpy.org).  The
   version from your distro's package manager should be fine for these two.
1. [Install Bazel](http://bazel.io/docs/install.html), which you may need to do
   from source.
1. `git clone --recurse-submodules https://github.com/tensorflow/tensorflow`
1. `cd tensorflow`
1. `./configure`
1. `bazel build -c opt --jobs=1 tensorflow:libtensorflow_c.so`

### Install the Vala bindings

1. Clone this repository and `cd` into it and:

``` bash
meson build --prefix <install prefix (/usr/local by default)>
ninja -C build
ninja -C build test
sudo ninja -C build install
```

### On Arch Linux

``` bash
yaourt -S tensorflow-vala
```

Which will drag `tensorflow` as a dependency.

## Disclaimer ##

It is worth noticing that the bindings have not been fully tested.
However, it's better to try to keep up with the C API as it is being built than to do it all at once.


[vala]:https://wiki.gnome.org/Projects/Vala
[tensorflow]:https://www.tensorflow.org/
[c_api]:https://www.tensorflow.org/code/tensorflow/c/c_api.h
[tf_bindings]:https://www.tensorflow.org/how_tos/language_bindings
[vlb]:https://wiki.gnome.org/Projects/Vala/LegacyBindings
[rust_bindings]: https://github.com/tensorflow/rust
