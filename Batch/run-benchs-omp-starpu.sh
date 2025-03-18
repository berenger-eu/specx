#!/bin/bash

#################################

# Function to check and remove core files in the current directory
remove_core_files() {
    # Check if any core files exist in the current directory
    if ls core.* 1> /dev/null 2>&1; then
        echo "Core files found. Removing..."
        rm core.*
        echo "Core files removed."
    else
        echo "No core files found."
    fi
}


function main(){
    RUN_DIR="/projets/schnaps/spetabaru-project/specx/build-ompstarpu-$SMPREFIX/"

    # Check if RUN_DIR exists
    if [ ! -d "$RUN_DIR" ]; then
        echo "make build dir"
        mkdir "$RUN_DIR"
    fi
    
    cd "$RUN_DIR"
    cmake .. -DSPECX_COMPILE_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$SMPREFIX -DCMAKE_BUILD_TYPE=RELEASE # RelWithDebInfo

    ## Redirect std and error to file 
    make -j > build.log 2>&1

    # To ensure we are in the right directory we test
    # that CMakeCache.txt file and Examples directory exist.
    if [ ! -f "$RUN_DIR/CMakeCache.txt" ] || [ ! -d "$RUN_DIR/Examples" ]; then
        echo "Please PREFIX variable should point to the SMPREFIX version to make a valid RUN_DIR, $RUN_DIR"
        return 1
    fi

    # Check if gemm cholesky axpy and particles-simu exist
    if [ ! -f "$RUN_DIR/Benchmark/particles_openmp/particles-simu-omp" ] || [ ! -f "$RUN_DIR/Benchmark/particles_starpu/particles-simu-starpu" ] || [ ! -f "$RUN_DIR/Benchmark/particles/particles-simu" ]; then
        echo "Please make sure that the benchmarks are built in $RUN_DIR"
        return 1
    fi
    
    # Create a directory to store the results
    # with the format results_[date]_[time]
    results_dir="results-$PREFIX-$(date +%Y%m%d_%H%M%S)"
    mkdir $results_dir

    echo "Running benchmarks, storing results in $results_dir"

    # Run the benchmarks
    NB_LOOPS=10

    # Particles
    "$RUN_DIR/Benchmark/particles/particles-simu" --lp=$NB_LOOPS --minp=500 --maxp=512 --minnbgroups=128 --maxnbgroups=256 --od="$results_dir" >> "$results_dir/output_particles.txt"
    remove_core_files
    
    # Particles omp
    "$RUN_DIR/Benchmark/particles_openmp/particles-simu-omp" --lp=$NB_LOOPS --minp=500 --maxp=512 --minnbgroups=128 --maxnbgroups=256 --od="$results_dir" >> "$results_dir/output_particles_omp.txt"
    remove_core_files
    
    # Particles starpu lws
    STARPU_SCHED=lws "$RUN_DIR/Benchmark/particles_starpu/particles-simu-starpu" --lp=$NB_LOOPS --minp=500 --maxp=512 --minnbgroups=128 --maxnbgroups=256 --od="$results_dir" >> "$results_dir/output_particles_starpu_lws.txt"
    remove_core_files
    
    # Particles starpu dmda
    STARPU_SCHED=dmda "$RUN_DIR/Benchmark/particles_starpu/particles-simu-starpu" --lp=$NB_LOOPS --minp=500 --maxp=512 --minnbgroups=128 --maxnbgroups=256 --od="$results_dir" >> "$results_dir/output_particles_starpu_dmda.txt"
    remove_core_files
}

module load tools/git/2.36.0 build/cmake/3.21.3 linalg/mkl/2020_update4 #  compiler/cuda/12.3 compiler/gcc/10.2.0
module load compiler/cuda/11.7 compiler/gcc
module load hardware/hwloc runtime/starpu/1.4.4/mpi-cuda

# module remove compiler/cuda/11.7
# module load compiler/cuda/12.3

module li

export STARPU_LIB_DIR="$STARPU_LIB"
export STARPU_INC_DIR="$STARPU_INC" 

ulimit -c 0

main ;

