#!/bin/bash

#################################

function main(){
    RUN_DIR="/projets/schnaps/spetabaru-project/specx/build-$SMPREFIX/"

    # Check if RUN_DIR exists
    if [ ! -d "$RUN_DIR" ]; then
        echo "make build dir"
        mkdir "$RUN_DIR"
    fi
    
    cd "$RUN_DIR"
    cmake .. -DSPECX_COMPILE_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$SMPREFIX -DCMAKE_BUILD_TYPE=RelWithDebInfo # RELEASE

    ## Redirect std and error to file 
    make -j > build.log 2>&1

    # To ensure we are in the right directory we test
    # that CMakeCache.txt file and Examples directory exist.
    if [ ! -f "$RUN_DIR/CMakeCache.txt" ] || [ ! -d "$RUN_DIR/Examples" ]; then
        echo "Please PREFIX variable should point to the SMPREFIX version to make a valid RUN_DIR, $RUN_DIR"
        return 1
    fi

    # Check if gemm cholesky axpy and particles-simu exist
    if [ ! -f "$RUN_DIR/Benchmark/axpy/axpy" ] || [ ! -f "$RUN_DIR/Benchmark/cholesky_gemm/cholesky" ] || [ ! -f "$RUN_DIR/Benchmark/cholesky_gemm/gemm" ] || [ ! -f "$RUN_DIR/Benchmark/particles/particles-simu" ]; then
        echo "Please make sure that the benchmarks are built in $RUN_DIR"
        return 1
    fi
    
    cd "/home/bramas/spetabaru-project/results/"

    # Create a directory to store the results
    # with the format results_[date]_[time]
    results_dir="results-$PREFIX-$(date +%Y%m%d_%H%M%S)"
    mkdir $results_dir

    echo "Running benchmarks, storing results in $results_dir"

    # Run the benchmarks
    NB_LOOPS=10

    # AXPY
    # TODO "$RUN_DIR/Benchmark/axpy/axpy" --lp=$NB_LOOPS --minnbb=16 --maxnbb=256 --minbs=128 --maxbs=65536 --gputh=256 --od="$results_dir" >> "$results_dir/output_axpy.txt"

    # Cholesky/gemm
    "$RUN_DIR/Benchmark/cholesky_gemm/cholesky" --lp=$NB_LOOPS --minms=4096 --maxms=8192 --minbs=128 --maxbs=512 --od="$results_dir" >> "$results_dir/output_cholesky.txt"
    "$RUN_DIR/Benchmark/cholesky_gemm/gemm" --lp=$NB_LOOPS --minms=4096 --maxms=8192 --minbs=128 --maxbs=512 --od="$results_dir" >> "$results_dir/output_gemm.txt"

    # Particles
    "$RUN_DIR/Benchmark/particles/particles-simu" --lp=$NB_LOOPS --minp=500 --maxp=8000 --minnbgroups=128 --maxnbgroups=512 --od="$results_dir" >> "$results_dir/output_particles.txt"
}

module load tools/git/2.36.0 compiler/cuda/12.3 compiler/gcc/10.2.0 build/cmake/3.21.3 linalg/mkl/2020_update4

module li

main ;

