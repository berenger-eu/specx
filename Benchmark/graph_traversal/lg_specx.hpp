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

    for(size_t i=0; i<graph.length(); i++){
        Node& n = graph.node_at(graph.level()-1, i);
        tg.task(SpWrite(n), [](Node& nparam){
            nparam.mark();
        });
    }

    for(int l=graph.level()-2; l>=0 ; l--){
        for(size_t i=0; i<graph.length(); i++){
            Node& n = graph.node_at(l, i);
            if(n._out_edges.size() && n._in_edges.size()){
                tg.task(SpWrite(n), SpWriteArray(graph.nodes_at(l+1).data(),SpArrayView(n._out_edges)),
                        SpReadArray(graph.nodes_at(l-1).data(),SpArrayView(n._in_edges)),
                        [](Node& nparam, [[maybe_unused]] SpArrayAccessor<Node>& outNodes,
                           [[maybe_unused]] const SpArrayAccessor<const Node>& inNodes){
                    nparam.mark();
                });
            }
            else if(n._out_edges.size()){
                tg.task(SpWrite(n), SpWriteArray(graph.nodes_at(l+1).data(),SpArrayView(n._out_edges)),
                        [](Node& nparam, [[maybe_unused]] SpArrayAccessor<Node>& outNodes){
                    nparam.mark();
                });
            }
            else if(n._in_edges.size()){
                tg.task(SpWrite(n), SpReadArray(graph.nodes_at(l-1).data(),SpArrayView(n._in_edges)),
                        [](Node& nparam, [[maybe_unused]] const SpArrayAccessor<const Node>& inNodes){
                    nparam.mark();
                });
            }
            else{
                tg.task(SpWrite(n), [](Node& nparam){
                    nparam.mark();
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

