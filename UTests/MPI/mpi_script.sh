#!/bin/bash
echo "mpirun -np $1 $2"
mpirun -np "$1" "$2"
