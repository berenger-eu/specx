// This code has been taken from the taskflow repository for reproducibility:
// https://github.com/taskflow/taskflow
#include <iostream>
#include <chrono>
#include <omp.h>

#include "levelgraph.hpp"

void traverse_regular_graph_omp(LevelGraph& graph, unsigned num_threads){

  omp_set_num_threads(num_threads);

  #pragma omp parallel
  {
    #pragma omp single
    {
      for(size_t l=0; l<graph.level(); l++){
        for(size_t i=0; i<graph.length(); i++){
          Node& n = graph.node_at(l, i);
          size_t out_edge_num = n._out_edges.size();
          size_t in_edge_num = n._in_edges.size();

          switch(in_edge_num){

            case(0):{

              switch(out_edge_num){

                case(1):{
                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  #pragma omp task depend(inout: out0, n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  #pragma omp task depend(inout: out0, out1, n)
                  { n.mark(); }
                  break;
                }

                case(3):{
                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  #pragma omp task depend(inout: out0, out1, out2, n)
                  { n.mark(); }
                  break;
                }

                case(4):{
                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  Node& out3 = graph.node_at(l+1, n._out_edges[3]);
                  #pragma omp task depend(inout: out0, out1, out2, out3, n)
                  { n.mark(); }
                  break;
                }

              }
              break;
            }


            case(1):{
              Node& in0 = graph.node_at(l-1, n._in_edges[0]);

              switch(out_edge_num){

                case(1):{
                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  #pragma omp task depend(in: in0) depend(inout: out0, n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  #pragma omp task depend(in: in0) depend(inout: out0, out1, n)
                  { n.mark(); }
                  break;
                }

                case(3):{

                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  #pragma omp task depend(in: in0) depend(inout: out0, out1, out2, n)
                  { n.mark(); }
                  break;
                }

                case(4):{

                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  Node& out3 = graph.node_at(l+1, n._out_edges[3]);
                  #pragma omp task depend(in: in0) depend(inout: out0, out1, out2, out3, n)
                  { n.mark(); }
                  break;
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
                  #pragma omp task depend(in: in0, in1) depend(inout: out0, n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  #pragma omp task depend(in: in0, in1) depend(inout: out0, out1, n)
                  { n.mark(); }
                  break;
                }

                case(3):{

                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  #pragma omp task depend(in: in0, in1) depend(inout: out0, out1, out2, n)
                  { n.mark(); }
                  break;

                }

                case(4):{

                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  Node& out3 = graph.node_at(l+1, n._out_edges[3]);
                  #pragma omp task depend(in: in0, in1) depend(inout: out0, out1, out2, out3, n)
                  { n.mark(); }
                  break;
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
                  #pragma omp task depend(in: in0, in1, in2) depend(inout: out0, n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  #pragma omp task depend(in: in0, in1, in2) depend(inout: out0, out1, n)
                  { n.mark(); }
                  break;

                }

                case(3):{

                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  #pragma omp task depend(in: in0, in1, in2) depend(inout: out0, out1, out2, n)
                  { n.mark(); }
                  break;
                }

                case(4):{

                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  Node& out3 = graph.node_at(l+1, n._out_edges[3]);
                  #pragma omp task depend(in: in0, in1, in2) depend(inout: out0, out1, out2, out3, n)
                  { n.mark(); }
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
                  #pragma omp task depend(in: in0, in1, in2, in3) depend(inout: out0, n)
                  { n.mark(); }
                  break;
                }

                case(2):{
                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  #pragma omp task depend(in: in0, in1, in2, in3) depend(inout: out0, out1, n)
                  { n.mark(); }
                  break;

                }

                case(3):{

                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  #pragma omp task depend(in: in0, in1, in2, in3) depend(inout: out0, out1, out2, n)
                  { n.mark(); }
                  break;

                }

                case(4):{

                  Node& out0 = graph.node_at(l+1, n._out_edges[0]);
                  Node& out1 = graph.node_at(l+1, n._out_edges[1]);
                  Node& out2 = graph.node_at(l+1, n._out_edges[2]);
                  Node& out3 = graph.node_at(l+1, n._out_edges[3]);
                  #pragma omp task depend(in: in0, in1, in2, in3) depend(inout: out0, out1, out2, out3, n)
                  { n.mark(); }
                  break;
                }
              }
            break;
            }
          }
        }
      }
    }
  }
}

std::chrono::microseconds measure_time_omp(LevelGraph& graph, unsigned num_threads){
  auto beg = std::chrono::high_resolution_clock::now();
  traverse_regular_graph_omp(graph, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


