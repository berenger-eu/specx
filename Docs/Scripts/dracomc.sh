#!/usr/bin/env bash
#SBATCH -J specx
#SBATCH -D ./
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# for OpenMP:
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH -p short
#SBATCH --mail-type=ALL --mail-user=berenger.bramas@mpcdf.mpg.de

module load cmake
module load gcc/7.2

cd /u/bbramas/specx/Build/test

outputdir="results"
cptres=0
while [ -d "$outputdir" ] ; do
    cptres=$((cptres+1))
    outputdir="results-"$cptres
done

mkdir "$outputdir"

for th in 1 5 10 ; do
    for loops in 1 5 10 50 100 ; do
        export NBTHREADS=$th
        export NBLOOPS=$loops

        outputdirrun="$outputdir/run-$NBTHREADS-$NBLOOPS"

        mkdir "$outputdirrun"

        ./mc > "$outputdirrun/output.txt" 2>&1
        
        mv /tmp/spec_without_collision* /tmp/nospec_without_collision.* "$outputdirrun"
    done
done
