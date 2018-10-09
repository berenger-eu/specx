#!/usr/bin/env bash
#SBATCH -J spetabaru
#SBATCH -D ./
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# for OpenMP:
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH -p short
#SBATCH --mail-type=ALL --mail-user=berenger.bramas@mpcdf.mpg.de

module load cmake
module load gcc/7.2

cd /u/bbramas/spetabaru/Build/testremc

outputdir="results"
cptres=0
while [ -d "$outputdir" ] ; do
    cptres=$((cptres+1))
    outputdir="results-"$cptres
done

mkdir "$outputdir"



for loops in 5 10 50 100 ; do

    th=1
    export NBTHREADS=$th
    export NBLOOPS=$loops

    outputdirrun="$outputdir/run-$NBTHREADS-$NBLOOPS"

    mkdir "$outputdirrun"

    ./remc > "$outputdirrun/output.txt" 2>&1

    mv /tmp/remc* "$outputdirrun"

    for th in 5 10 20 30 ; do
        export NBTHREADS=$th
        export NBLOOPS=$loops

        outputdirrun="$outputdir/run-$NBTHREADS-$NBLOOPS"

        mkdir "$outputdirrun"

        ./remcnoseq > "$outputdirrun/output.txt" 2>&1
        
        mv /tmp/remc* "$outputdirrun"
    done
done
