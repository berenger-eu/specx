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


void traverse_level_graph_specx2(LevelGraph& graph, unsigned num_threads){
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
            size_t out_edge_num = n._out_edges.size();
            size_t in_edge_num = n._in_edges.size();

            switch(in_edge_num){

            case(1):{
                Node& in0 = graph.node_at(l-1, n._in_edges[0]);

                switch(out_edge_num){
                case(1):{
                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    tg.task(SpWrite(n),
                            SpWrite(out0),
                            SpRead(in0),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0,
                            [[maybe_unused]] const Node& nin0){
                        nparam.mark();
                    });
                    break;
                }

                case(2):{
                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1),
                            SpRead(in0),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1,
                            [[maybe_unused]] const Node& nin0){
                        nparam.mark();
                    });
                    break;

                }

                case(3):{

                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    Node& out2 = graph.node_at(l+1, n._out_edges[2]);

                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1), SpWrite(out2),
                            SpRead(in0),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1, [[maybe_unused]] Node& nout2,
                            [[maybe_unused]] const Node& nin0){
                        nparam.mark();
                    });
                }
                }
                break;
            }


            case(2):{

                Node& in0 = graph.node_at(l-1, n._in_edges[0]);
                Node& in1 = graph.node_at(l-1, n._in_edges[1]);

                switch(out_edge_num){
                case(1):{
                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    tg.task(SpWrite(n),
                            SpWrite(out0),
                            SpRead(in0), SpRead(in1),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1){
                        nparam.mark();
                    });
                    break;
                }

                case(2):{
                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1),
                            SpRead(in0), SpRead(in1),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1){
                        nparam.mark();
                    });
                    break;

                }

                case(3):{

                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    Node& out2 = graph.node_at(l+1, n._out_edges[2]);

                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1), SpWrite(out2),
                            SpRead(in0), SpRead(in1),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1, [[maybe_unused]] Node& nout2,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1){
                        nparam.mark();
                    });
                }
                }
                break;
            }


            case(3):{

                Node& in0 = graph.node_at(l-1, n._in_edges[0]);
                Node& in1 = graph.node_at(l-1, n._in_edges[1]);
                Node& in2 = graph.node_at(l-1, n._in_edges[2]);

                switch(out_edge_num){

                case(1):{
                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    tg.task(SpWrite(n),
                            SpWrite(out0),
                            SpRead(in0), SpRead(in1), SpRead(in2),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1, [[maybe_unused]] const Node& nin2){
                        nparam.mark();
                    });
                    break;
                }

                case(2):{
                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1),
                            SpRead(in0), SpRead(in1), SpRead(in2),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1, [[maybe_unused]] const Node& nin2){
                        nparam.mark();
                    });
                    break;

                }

                case(3):{

                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    Node& out2 = graph.node_at(l+1, n._out_edges[2]);

                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1), SpWrite(out2),
                            SpRead(in0), SpRead(in1), SpRead(in2),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1, [[maybe_unused]] Node& nout2,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1, [[maybe_unused]] const Node& nin2){
                        nparam.mark();
                    });
                    break;

                }

                case(4):{

                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                    Node& out3 = graph.node_at(l+1, n._out_edges[3]);

                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1), SpWrite(out2), SpWrite(out3),
                            SpRead(in0), SpRead(in1), SpRead(in2),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1, [[maybe_unused]] Node& nout2, [[maybe_unused]] Node& nout3,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1, [[maybe_unused]] const Node& nin2){
                        nparam.mark();
                    });
                    break;
                }
                }
                break;

            }


            case(4):{

                Node& in0 = graph.node_at(l-1, n._in_edges[0]);
                Node& in1 = graph.node_at(l-1, n._in_edges[1]);
                Node& in2 = graph.node_at(l-1, n._in_edges[2]);
                Node& in3 = graph.node_at(l-1, n._in_edges[3]);

                switch(out_edge_num){

                case(1):{
                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    tg.task(SpWrite(n),
                            SpWrite(out0),
                            SpRead(in0), SpRead(in1), SpRead(in2), SpRead(in3),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1, [[maybe_unused]] const Node& nin2, [[maybe_unused]] const Node& nin3){
                        nparam.mark();
                    });
                    break;
                }

                case(2):{
                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1),
                            SpRead(in0), SpRead(in1), SpRead(in2), SpRead(in3),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1, [[maybe_unused]] const Node& nin2, [[maybe_unused]] const Node& nin3){
                        nparam.mark();
                    });
                    break;

                }

                case(3):{

                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    Node& out2 = graph.node_at(l+1, n._out_edges[2]);

                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1), SpWrite(out2),
                            SpRead(in0), SpRead(in1), SpRead(in2), SpRead(in3),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1, [[maybe_unused]] Node& nout2,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1, [[maybe_unused]] const Node& nin2, [[maybe_unused]] const Node& nin3){
                        nparam.mark();
                    });
                    break;

                }

                case(4):{

                    Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                    Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                    Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                    Node& out3 = graph.node_at(l+1, n._out_edges[3]);

                    tg.task(SpWrite(n),
                            SpWrite(out0), SpWrite(out1), SpWrite(out2), SpWrite(out3),
                            SpRead(in0), SpRead(in1), SpRead(in2), SpRead(in3),
                            [](Node& nparam,
                            [[maybe_unused]] Node& nout0, [[maybe_unused]] Node& nout1, [[maybe_unused]] Node& nout2, [[maybe_unused]] Node& nout3,
                            [[maybe_unused]] const Node& nin0, [[maybe_unused]] const Node& nin1, [[maybe_unused]] const Node& nin2, [[maybe_unused]] const Node& nin3){
                        nparam.mark();
                    });
                    break;
                }
                }
                break;
            }
            }
        }
    }

    tg.waitAllTasks();
}

std::chrono::microseconds measure_time_specx2(LevelGraph& graph, unsigned num_threads){
    auto beg = std::chrono::high_resolution_clock::now();
    traverse_level_graph_specx2(graph, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

