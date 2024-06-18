#!/bin/bash



#################################
 
function main(){
    # To ensure we are in the right directory we test
    # that CMakeCache.txt file and Examples directory exist.
    if [ ! -f CMakeCache.txt ] || [ ! -d Examples ]; then
        echo "Please run this script from the build directory."
        return 1
    fi

    # Create a directory to store the results
    # with the format results_[date]_[time]
    results_dir="results_$(date +%Y%m%d_%H%M%S)"
    mkdir $results_dir

    echo "Running benchmarks, storing results in $results_dir"

    # Run the benchmarks
    NB_LOOPS=10

    # AXPY
    ./Benchmark/axpy/axpy --lp=$NB_LOOPS --minnbb=16 --maxnbb=256 --minbs=128 --maxbs=65536 --cuth=256 --od="$results_dir"

    # Cholesky/gemm
    ./Benchmark/cholesky_gemm/cholesky --lp=$NB_LOOPS --minms=4096 --maxms=16384 --minbs=128 --maxbs=512 --od="$results_dir"
    ./Benchmark/cholesky_gemm/gemm --lp=$NB_LOOPS --minms=4096 --maxms=16384 --minbs=128 --maxbs=512 --od="$results_dir"

    # Particles
    ./Benchmark/particles/particles-simu --lp=$NB_LOOPS --minp=5000 --maxp=100000 --minnbgroups=128 --maxnbgroups=512 --od="$results_dir"
}



