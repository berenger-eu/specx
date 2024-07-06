#!/bin/bash

#################################
 
function main(){
    # To ensure we are in the right directory we test
    # that CMakeCache.txt file and Examples directory exist.
    if [ ! -f $PREFIX/CMakeCache.txt ] || [ ! -d $PREFIX/Examples ]; then
        echo "Please PREFIX variable should point to the build dir"
        return 1
    fi
    
    cd /home/bramas/spetabaru-project/results/

    # Create a directory to store the results
    # with the format results_[date]_[time]
    results_dir="results_$(date +%Y%m%d_%H%M%S)"
    mkdir $results_dir

    echo "Running benchmarks, storing results in $results_dir"

    # Run the benchmarks
    NB_LOOPS=10

    # AXPY
    ./Benchmark/axpy/axpy --lp=$NB_LOOPS --minnbb=16 --maxnbb=256 --minbs=128 --maxbs=65536 --gputh=256 --od="$results_dir"

    # Cholesky/gemm
    ./Benchmark/cholesky_gemm/cholesky --lp=$NB_LOOPS --minms=4096 --maxms=16384 --minbs=128 --maxbs=512 --od="$results_dir"
    ./Benchmark/cholesky_gemm/gemm --lp=$NB_LOOPS --minms=4096 --maxms=16384 --minbs=128 --maxbs=512 --od="$results_dir"

    # Particles
    ./Benchmark/particles/particles-simu --lp=$NB_LOOPS --minp=500 --maxp=10000 --minnbgroups=128 --maxnbgroups=512 --od="$results_dir"
}

module load compiler/cuda/12.3 compiler/gcc/10.2.0 build/cmake/3.21.3 linalg/mkl/2020_update4

module li

main ;

