[![pipeline status](https://gitlab.inria.fr/bramas/spetabaru/badges/master/pipeline.svg)](https://gitlab.inria.fr/bramas/spetabaru/commits/master)
[![coverage report](https://gitlab.inria.fr/bramas/spetabaru/badges/master/coverage.svg)](https://gitlab.inria.fr/bramas/spetabaru/commits/master)


# Introduction
SPETABARU is a task-based runtime system, which is
capable of executing tasks in advance if some others are not certain to modify
the data.

The project was originally designed for Monte Carlo and
Replica Exchange Monte Carlo (RMC/Parallel tempering).

This is an on-going project under development.

# Installation
SPETABARU requires an out of source tree build.
1. First create a new directory outside of SPETABARU's source tree with mkdir <dir_name>
2. cd into the newly created directory.
3. Create a subdirectory with a name of your choice
4. Run cmake -DCMAKE_INSTALL_PREFIX=<path_to_directory_created_in_step_3> ..
5. Run make or make -j <number_of_commands_to_run_simultaneously>
6. Run make install

If you want to install the static SPETABARU library and its header files to your default user
library location you can remove the DCMAKE_INSTALL_PREFIX variable setting flag in step 4.

If you still prefer to create the build directory inside SPETABARU's source tree, we recommend
to call it "build" as this name has been explicitly marked as to be ignored by Git.

# Examples

Several examples are given in the `Examples` directory.


# Support

Please leave an issue on the SPETABARU repository:
https://gitlab.inria.fr/bramas/spetabaru

# Citing

You can refer to the paper available at https://peerj.com/articles/cs-183/
This document is also describing the current models.
