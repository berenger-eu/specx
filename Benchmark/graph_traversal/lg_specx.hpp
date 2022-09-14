// This code has been taken from the taskflow repository for reproducibility:
// https://github.com/taskflow/taskflow
#include <iostream>
#include <chrono>

#include "Utils/SpArrayView.hpp"
#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"

#include "levelgraph.hpp"

void traverse_level_graph_specx(LevelGraph& graph, unsigned num_threads){
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(num_threads));
    SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
    tg.computeOn(ce);

    for(size_t l=0; l<graph.level(); l++){
        for(size_t i=0; i<graph.length(); i++){
            Node& n = graph.node_at(l, i);

            if( n._in_edges.size() && n._out_edges.size()){
                tg.task(SpReadArray(n._in_edges.data(),SpArrayView(n._in_edges.size())),
                             SpWriteArray(n._out_edges.data(),SpArrayView(n._out_edges.size())),
                             [nptr=&n]([[maybe_unused]] const SpArrayAccessor<const std::pair<int, int>>& inNodes,
                                       [[maybe_unused]] SpArrayAccessor<int>& outNodes){
                    nptr->mark();
                });
            }
            else if(n._in_edges.size()){
                tg.task(SpReadArray(n._in_edges.data(),SpArrayView(n._in_edges.size())),
                             [nptr=&n]([[maybe_unused]] const SpArrayAccessor<const std::pair<int, int>>& inNodes){
                    nptr->mark();
                });
            }
            else{
                tg.task(SpWriteArray(n._out_edges.data(),SpArrayView(n._out_edges.size())),
                             [nptr=&n]([[maybe_unused]] SpArrayAccessor<int>& outNodes){
                    nptr->mark();
                });
            }
        }
    }

    tg.waitAllTasks();
}

std::chrono::microseconds measure_time_specx(LevelGraph& graph, unsigned num_threads){
    auto beg = std::chrono::high_resolution_clock::now();
    traverse_level_graph_specx(graph, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

