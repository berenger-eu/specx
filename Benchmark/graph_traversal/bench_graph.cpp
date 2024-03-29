// This code has been taken from the taskflow repository for reproducibility:
// https://github.com/taskflow/taskflow
// and updated to remove CLI11 use by clsimple.
#include "levelgraph.hpp"
#ifdef _OPENMP
#include "lg_omp.hpp"
#endif
#include "lg_specx.hpp"
#include <clsimple.hpp>

#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {

  CLsimple args("Graph Traversal", argc, argv);
  
  args.addParameterNoArg({"help"}, "help");

  unsigned num_threads;
  args.addParameter<unsigned>({"t" ,"num_threads"}, "number of threads", num_threads, 1);

  unsigned num_rounds;
  args.addParameter<unsigned>({"r" ,"num_rounds"}, "number of rounds", num_rounds, 1);

  std::string model;
  args.addParameter<std::string>({"m" ,"model"}, "model name specx|omp|specx2", model, "specx");

  unsigned size;
  args.addParameter<unsigned>({"s" ,"size"}, "size", size, 256);

  args.parse();

  if(!args.isValid() || args.hasKey("help")
        || !(model == "specx"
#ifdef _OPENMP
        	 || model == "omp"
#endif        	 
        	 || model == "specx2")){
    // Print the help
    args.printHelp(std::cout);
    return -1;
  }

  std::cout << "model=" << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds=" << num_rounds << ' '
            << std::endl;

  std::cout << std::setw(12) << "|V|+|E|"
            << std::setw(12) << "Runtime"
             << '\n';

  for(int i=1; i<=size; i += 15) {

    double runtime {0.0};

    LevelGraph graph(i, i);

    for(unsigned j=0; j<num_rounds; ++j) {
      if(model == "specx") {
        runtime += measure_time_specx(graph, num_threads).count();
      }
#ifdef _OPENMP
      else if(model == "omp") {
        runtime += measure_time_omp(graph, num_threads).count();
      }
#endif
      else if(model == "specx2") {
        runtime += measure_time_specx2(graph, num_threads).count();
      }
      graph.clear_graph();
    }

    std::cout << std::setw(12) << graph.graph_size()
              << std::setw(12) << runtime / num_rounds / 1e3
              << std::endl;

  }
}

