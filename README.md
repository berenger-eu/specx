[![pipeline status](https://gitlab.inria.fr/bramas/specx/badges/master/pipeline.svg)](https://gitlab.inria.fr/bramas/specx/commits/master)
[![coverage report](https://gitlab.inria.fr/bramas/specx/badges/master/coverage.svg)](https://bramas.gitlabpages.inria.fr/specx/)


This work has been funded by the Inria ADT project SPETABARU-H, and
the ANR National project AUTOSPEC (ANR-21-CE25-0009).

# Introduction
SPECX is a task-based runtime system.
It shares several similarities with StarPU but is written in modern C++.
It also supports speculative execution, which is
the capability of executing tasks in advance if some others are not certain to modify
the data.

The project was originally designed for Monte Carlo and
Replica Exchange Monte Carlo (RMC/Parallel tempering).

This is an on-going project under development.

# GitHub Actions

[![CI](https://github.com/feelpp/specx/actions/workflows/ci.yml/badge.svg)](https://github.com/feelpp/specx/actions/workflows/ci.yml)

# Installation requirements

- A C++20 toolchain for CPU and HIP builds; CUDA builds currently use C++17.
- CMake 3.20 or newer and a platform with POSIX threads.
- Optional: CUDA, HIP, or MPI development tools for the corresponding backend.
  CUDA and HIP cannot be enabled together.

# Build, test, and install

From the repository root, a CPU-only Debug build can be configured and tested with:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSPECX_COMPILE_WITH_CUDA=OFF \
  -DSPECX_COMPILE_WITH_HIP=OFF \
  -DSPECX_COMPILE_WITH_MPI=OFF \
  -DSPECX_BUILD_BENCHMARK=OFF
cmake --build build --parallel 2
OMP_NUM_THREADS=4 ctest --test-dir build --output-on-failure --timeout 90
```

CUDA is enabled by default when its compiler is detected, so explicitly disable it
for a CPU-only build. Use `Release` for optimized builds or `RelWithDebInfo` for
optimized builds with debug symbols. Adjust build parallelism to available memory.

To install the library and headers to a user-selected prefix:

```sh
cmake --install build --prefix "$HOME/.local"
```

Useful CMake options:

| Option | Purpose |
| --- | --- |
| `SPECX_JUST_LIB=ON` | Build only the library; the default when included as a subproject. |
| `SPECX_BUILD_UTESTS=OFF` | Disable unit tests. |
| `SPECX_BUILD_EXAMPLES=OFF` | Disable examples. |
| `SPECX_BUILD_BENCHMARK=OFF` | Disable benchmarks and their optional dependencies. |
| `SPECX_DO_INSTALL=OFF` | Disable installation of the library and generated configuration. |
| `SPECX_USE_DEBUG_PRINT=ON` | Compile support for debug logging; enabled by default. |

See [Docs/Testing.md](./Docs/Testing.md) for regression tests and sanitizer checks.

# Examples

Several examples are given in the `Examples` and `UTests` directories.
Setting the environment variable `SPECX_DEBUG_PRINT` to `TRUE` will enable debug information output. 

# Support
Please leave an issue on the SPECX repository:
https://gitlab.inria.fr/bramas/specx

# Citing

You can refer to the paper available at https://peerj.com/articles/cs-183/
This document is also describing the current models.

# Quick overview example
```cpp
#include <random>
#include <cassert>

#include "Data/SpDataAccessMode.hpp"
#include "Task/SpPriority.hpp"
#include "Task/SpProbability.hpp"
#include "Legacy/SpRuntime.hpp"

const size_t seedSpeculationSuccess = 42;
const size_t seedSpeculationFailure = 0;
const size_t seed = seedSpeculationSuccess;

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]){
    // First we instantiate a runtime object and we specify that the
    // runtime should use speculation model 2.
    SpRuntime<SpSpeculativeModel::SP_MODEL_2> runtime;
    
    // Next we set a predicate that will be called by the runtime each
    // time a speculative task becomes ready to run. It is used to
    // decide if the speculative task should be allowed to run.
    runtime.setSpeculationTest(
    []([[maybe_unused]] const int nbReadyTasks,
    [[maybe_unused]] const SpProbability& meanProbability) -> bool {
        return true; // Here we always return true, this basically means
                     // that we always allow speculative tasks to run
                     // regardless of runtime conditions.
    });

    int a = 41, b = 0, c = 0;

    // We create our first task. We are specifying that the task will be
    // reading from a. The task will call the lambda given as a last
    // argument to the call. The return value of the task is the return
    // value of the lambda.  
    auto task1 = runtime.task(SpRead(a), [](const int& inA) -> int {
        return inA + 1;
    });
    
    // Here we set a custom name for the task.
    task1.setTaskName("First-task");
    
    // Here we wait until task1 is finished and we retrieve its return
    // value. 
    b = task1.getValue();
    
    // Next we create a potential task, i.e. a task which might write to
    // some data.
    // In this case the task may write to "a" with a probability of 0.5.
    // Subsequent tasks will be allowed to speculate over this task.
    // The task returns a boolean to inform the runtime of whether or 
    // not it has written to its maybe-write data dependency a.
    std::mt19937_64 mtEngine(seed);
    std::uniform_real_distribution<double> dis01(0,1);
    
    runtime.task(SpPriority(0), SpProbability(0.5), SpRead(b),
    SpPotentialWrite(a),
    [dis01, mtEngine] (const int &inB, int &inA) mutable -> bool{
        double val = dis01(mtEngine);
        
        if(inB == 42  && val < 0.5) {
            inA = 43;
            return true;
        }
        
        return false;
        
    }).setTaskName("Second-task");
    
    // We create a final normal task that reads from a and writes to c.
    // The task reads from a so there should be a strict write -> read
    // dependency between the second and the final task but since the
    // second task may not always write to a, the runtime will try to
    // execute a speculative version of the final task in parallel
    // with the second task in case the second task doesn't write to a.
    runtime.task(SpRead(a), SpWrite(c), [] (const int &inA, int &inC) {
        if(inA == 41) {
            inC = 1;
        } else {
            inC = 2;
        }
    }).setTaskName("Final-task");

    // We wait for all tasks to finish
    runtime.waitAllTasks();
    
    // We make all runtime threads exit
    runtime.stopAllThreads();
    
    assert((a == 41 || a == 43) && b == 42 && (c == 1 || c == 2)
            && "Try again!");
    
    // We generate the task graph corresponding to the execution 
    runtime.generateDot("/tmp/getting-started.dot", true);
    
    // We generate an Svg trace of the execution
    runtime.generateTrace("/tmp/getting-started.svg");
    
    return 0;
}
```
You can find the source file for this example at [Examples/Basic/getting-started.cpp](./Examples/Basic/getting-started.cpp)

| Execution task graph (speculation succeeded) | Execution task graph (speculation failed) |
:---------------------------------------------:|:------------------------------------------:
![Execution task graph (speculation succeded)](./Images/getting-started-success.png) | ![Execution task graph (speculation failed)](./Images/getting-started-failure.png)

# Detailed overview
## Workflow
Workflow in Specx mainly revolves around three interfaces:
- the runtime interface
- the data dependency interface
- the task viewer interface

## Task graphs and compute engines

`SpTaskGraph` owns tasks and their dependencies. `SpComputeEngine` owns workers and
the scheduler used to execute ready tasks. Several graphs can share one engine;
workers can also be transferred between engines. `Legacy/SpRuntime.hpp` provides
`SpRuntime`, a convenience wrapper that owns one graph and one engine.

A minimal CPU example using the graph and engine interfaces:

```cpp
#include <cassert>
#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"

int main() {
    int value = 0;
    SpComputeEngine engine(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
    SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> graph;
    graph.computeOn(engine);
    graph.task(SpWrite(value), [](int& output) { output = 42; });
    graph.waitAllTasks();
    assert(value == 42);
    engine.stopIfNotAlreadyStopped();
}
```

Keep dependency data alive until all tasks using it finish. Wait for each graph
before stopping its engine, and destroy graphs before the engine they use. An
engine stops its workers on destruction, but does not drain unfinished graphs.
An engine with no workers cannot make progress until workers are added.

The default CPU scheduler is `SpSimpleScheduler`, which uses LIFO order for ready
tasks and does not use priorities. Pass `std::make_unique<SpPrioScheduler>()` as the
second argument to the engine constructor when priorities are needed. Neither
scheduler changes the ordering imposed by data dependencies.

Worker management and `execOnWorkers` calls must be serialized by the application.
`execOnWorkers` runs its callback on each worker and waits for each invocation to
finish; callback side effects are visible when it returns. Call it from the
controlling thread, and keep callbacks non-throwing. Workers detached from an
engine remain started and can be added to another engine or safely destroyed.

## Runtime interface
The runtime's functionality is exposed through a class called `SpRuntime`. This class
provides task creation, task graph and trace generation facilities.

`SpRuntime` is templated on `SpSpeculativeModel`: `SP_MODEL_1`, `SP_MODEL_2`,
`SP_MODEL_3`, or `SP_NO_SPEC`. The default is `SP_MODEL_1`. Definitions are in
[Src/Speculation/SpSpeculativeModel.hpp](./Src/Speculation/SpSpeculativeModel.hpp).

### Main `SpRuntime` methods:

- SpRuntime(const int inNumThreads)

Each `SpRuntime` owns its task graph and compute engine. The explicit thread-count
constructor creates a CPU worker team; use a positive thread count. The default
CPU count comes from `OMP_NUM_THREADS`, then `SPECX_NB_CPU_THREADS`, then the
hardware concurrency reported by the standard library. In a CUDA build, the
no-argument constructor creates a combined CPU/CUDA team. Use the graph and
engine interfaces described above to share or transfer workers.

- auto task([optional] SpPriority inPriority, [optional] SpProbability inProbability,  
[optional] \<DataDependencyTy\> ddo..., \<CallableTy\> c) (1)
- auto task([optional] SpPriority inPriority, [optional] SpProbability inProbability,  
[optional] \<DataDependencyTy\> ddo..., SpCpu(\<CallableTy\> c1), [optional] SpCuda(\<CallableTy\> c2)) (2)

This method creates a new task and injects it into the runtime. It returns an object representing the newly created task. For more information
on how to interact with this object see `Task viewer interface` section below.
 
The inPriority parameter specifies a priority for the task. You can find the definition of the SpPriority class in
[Src/Task/SpPriority.hpp](./Src/Task/SpPriority.hpp).

The inProbability parameter is an object used to specify the probability with which the task may write to its maybe-written data dependencies. You can
find the definition of the SpProbability class in [Src/Task/SpProbability.hpp](./Src/Task/SpProbability.hpp).

Following the inProbability parameter is a list of data dependency objects (see section Data dependency interface below for a definition of these
objects). This list declares the data dependencies of the task. At the moment we only allow one type of data dependency to be declared for a given
data element and a data dependency declaration of a certain type for a particular data element should only appear once except for read and atomic
write dependencies. For example you can't have a read and a write dependency for the same data element (in this case you should only declare the
strongest dependency type which is write). Validity of the dependencies is checked at runtime. If you have declared two data dependencies on
expressions that are different but evaluate to the same data element the program will exit.  
The runtime distinguishes potential tasks (tasks that might not always write to their write data dependencies) from normal tasks (tasks
that always write to their write data dependencies). The information of whether the to be created task should be considered a potential task or
a normal task is infered from the DataDependency object list. In short, if the list contains a data dependency object of type SpPotentialWrite or
SpPotentialWriteArray (see data dependency section below) then the task will be considered a potential task. If not it will be considered a normal task.
The runtime will only try to speculate over potential tasks.

The last or two last argument(s) (depending on which overload the call gets resolved to) specifies(specify) (a) callable(s) embodying the code that
the task should execute. The callables can either be lambda expressions or functors.
The callable's function call operator must have as many parameters as there are data dependency objects in the data dependency object list. For CPU code,
scalar parameters should be lvalue references matching the corresponding data
dependency (`const` for reads); array dependencies use `SpArrayAccessor`. CUDA and
HIP callables instead receive `SpDeviceDataView` objects, as shown in the backend
sections below. The parameters should appear in the
same order as they appear in the data dependency list.  
Example:
```cpp
Type1 v1;
Type2 v2;

runtime.task(SpRead(v1), SpWrite(v2), [] (const Type1 &paramV1, Type2 &paramV2) {
    if(paramV1.test()) {
        paramV2.set(1);
    }else {
        paramV2.set(2);
    }
}); 
```
Parameters corresponding to a SpRead data dependency object should be declared const (paramV1 in the example given above).
Code inside the callable must be referring to parameter names rather than the original variable names. In the example given above, code in the lambda
body is referring to the names paramV1 and paramV2 to refer to v1 and v2 data values rather than v1 and v2. You should not capture v1 and v2 by reference
and work with v1 and v2 directly. You can however capture any variable that does not appear in the data dependency list and work with it directly.
The runtime will store addresses to the data elements appearing in the data dependency list and will take care of calling the callable with the
appropriate corresponding arguments. In the example given above, assuming the task call is the sole task call in the entire program, the runtime will take
the addresses of v1 and v2 (since these are the data elements that appear in the data dependency list) and when the task executes it will call the lambda
with arguments \*v1 and \*v2. Note that since Specx is a speculative task-based runtime system it will also happen that the callable gets called with
copies of the data elements (sampled at different points in time) rather then the original data elements.  
The callables for normal tasks can return any value.
The callables for potential tasks must however all return a boolean. This boolean is used to inform the runtime of whether the task has written
to its maybe-write data dependencies or not. The callable's code must appropriately return true or false depending on the situation. It should return
true if the task wrote to its maybe-write data dependencies and false otherwise.  
In overload (1) the callable is passed as is to the task call. It will implicitly be interpreted by the runtime as CPU code.  
In overload (2) the callable c1 is explictly tagged as CPU code by being wrapped inside a SpCpu object (see subsection on callable wrapper objects
in section Data dependency interface below). Overload (2) additionally permits the user to provide a GPU version of the code (in this case the callable
should be wrapped inside a SpCuda object). When both CPU and GPU versions of the code are provided, the Specx runtime will decide at runtime
which one of the two to execute.           

- void setSpeculationTest(std::function<bool(int,const SpProbability&)> inFormula)

This method sets a predicate function that will be called by the runtime whenever a speculative task is ready to be put in the queue of ready tasks
(i.e. all of its data dependencies are ready). The predicate is used to decide, based on runtime information, if the speculative task as well as
any of its dependent speculative tasks should be allowed to run. The predicate returns a boolean. A return value of true means the speculative task 
and all of its dependent speculative tasks are allowed to run. Reciprocally a return value of false means the speculative task and all of its dependent
speculative tasks should be disabled.

Note that while a speculative task might be allowed to run it doesn't necessarely mean that it will actually run. For a speculative task to 
actually run all the parent speculations over which it speculates must not have failed. It could be that in between the time the speculative task
was marked as allowed to run and the time where it is actually picked up by a thread for execution some of the parent speculations have failed and so it
won't get executed even though it was allowed to run as per result of the evaluation of the predicate in the past.  
      
The two arguments to the predicate are provided by the runtime. The first parameter corresponds to the number of tasks that were in the ready queue
at the time the predicate was called. The second parameter is a probability whose value is the mean of all the probabilities of all the dependent speculative
tasks of the speculative task that the predicate is being called for and the probability of the speculative task that the predicate is being
called for.
Based on these two parameters one can write its own custom logic to enable/disable speculative tasks. You could for example decide to disable a speculative
task if the mean probability goes beyond a certain threshhold (since it might not make much sense to continue speculating if the chances of failure are high).
The prototype of the predicate might change in the future as we might want to take into account additional or different data to perform the decision.

If no speculation test is set in the runtime, the default behavior is that a speculative task and all its dependent speculative tasks will only be enabled if
at the time the predicate is called no other tasks are ready to run.

- void waitAllTasks()

This method is a blocking call which waits until all the tasks that have been pushed to the runtime up to this point have finished.

- void waitRemain(const long int windowSize)

This method is a blocking call which waits until the number of still unprocessed tasks becomes less than or equal to windowSize.

- void stopAllThreads()

This method is a blocking call which causes the runtime threads to exit. The method expects all tasks to have already finished, therefore you should always
call waitAllTasks() before calling this method.

- int getNbThreads()

This method returns the size of the runtime thread pool (in number of threads).

- void generateDot(const std::string& outputFilename, bool printAccesses)

This method will generate the task graph corresponding to the execution in dot format.  
It will write its output to `outputFilename` path. The boolean `printAccesses` can be set to true if you want to print the memory accesses of the tasks
(only the memory accesses that were specified in their data dependency list will get printed) into the body of the tasks node. By default `printAccesses`
is set to false.  
Tasks names will be printed into the graph nodes. The default name will be output for each task unless another name was manually set by the user (see 
`Task viewer interface` section below). Speculative versions of tasks will have an apostrophe appended to their name. 
You can visualize the task graph in pdf format using the following command :

`dot -Tpdf -o <pdf_output_filename> <path_to_dot_output_file>`

The generateDot method should be called after having called waitAllTasks() and stopAllThreads().

- void generateTrace(const std::string& outputFilename, const bool showDependences)

This method will generate a trace of the execution (with timings and dependences) in svg format.  
The generateTrace method should only be called after having called waitAllTasks() and stopAllThreads().

## Data dependency interface
The data dependency interface forms a collection of objects that can be used to express data dependencies.  
It also provides wrapper objects that can be used to specify if a given callable should be considered CPU or GPU code.  
The definition of the classes for these objects can be found in [Src/Data/SpDataAccessMode.hpp](./Src/Data/SpDataAccessMode.hpp).

### Data dependency objects
Specifying data dependencies boils down to constructing the relevant data dependency objects from the data lvalues.

#### Scalar data
- SpRead(x)

Specifies a read dependency on x.  
Read requests are always fulfilled by default, i.e. a read request rr2 on data x immediately following another read request rr1 on data x does
not have to wait for rr1 to be fulfilled in order to be serviced.
Multiple successive read accesses will be performed in any order and/or at the same time.  
Reads are ordered by the runtime with respect to writes, maybe-writes, commutative writes and parallel writes.
The order is the order in which the data accesses have been requested at runtime. 
 
- SpWrite(x)

Specifies a write dependency on x indicating that the data x will be written to with 100\% certainty.  
Multiple successive write requests to given data x will be fulfilled one after the other in the order they were emitted in at
runtime.  
Writes are ordered by the runtime with respect to reads, writes, maybe-writes, commutative writes and parallel writes.
The order is the order in which the data accesses have been requested at runtime.

- SpPotentialWrite(x)

Specifies a maybe-write dependency indicating that the data x might be written to, i.e. it will not always be the case (writes might occur
with a certain probability).  
Multiple successive maybe-write requests to given data x will be fulfilled one after the other in the order they were emitted in at runtime.  
Maybe-writes are ordered by the runtime with respect to reads, writes, maybe-writes, commutative writes and parallel writes.
The order is the order in which the data accesses have been requested at runtime.


- SpCommutativeWrite(x)

Specifies a commutative write dependency on x, i.e. writes that can be performed in any order.  
Multiple successive commutative write requests will be fulfilled one after the other in any order : while a commutative write request cw1 on
data x is currently being serviced, all immediately following commutative write request on data x will be put on hold. When cw1 is released
one of the immediately following commutative write requests will be serviced. No order is enforced by the runtime as to which one gets serviced
next.
As an example, if two tasks commutative write to data x, the runtime does not enforce an order as to which tasks gets to write first. The two
tasks won't be able to execute in parallel however : while one of the two task is executing and writing to data x the other task won't be able
to execute because its commutative write dependency request won't be serviced until the first task has finished executing and has released its
commutative write dependency on x.  
Commutative writes are ordered by the runtime with respect to reads, writes, maybe-writes and parallel writes.
The order is the order in which the data accesses have been requested at runtime.

- SpParallelWrite(x)

Specifies a parallel write dependency on x. Multiple successive parallel write
requests may execute at the same time and in any order. The callable must provide
its own synchronization, for example with `std::atomic` or a mutex; this access
mode does not make ordinary C++ writes atomic.

Parallel writes are ordered with respect to reads, writes, potential writes, and
commutative writes in the order those accesses were submitted.

All data dependency constructors for scalar data should be given an lvalue as the argument.


#### Non scalar data
We also provide analogous contructors for aggregates of data values from arrays :
- SpReadArray(\<XTy\> \*x, \<ViewTy\> view)
- SpWriteArray(\<XTy\> \*x, \<ViewTy\> view)
- SpPotentialWriteArray(\<XTy\> \*x, \<ViewTy\> view)
- SpCommutativeWriteArray(\<XTy\> \*x, \<ViewTy\> view)
- SpParallelWriteArray(\<XTy\> \*x, \<ViewTy\> view)

`x` should be a pointer to a contiguous buffer (the array).\
`view` should be an object representing the collection of specific indices of the array elements that are affected by the dependency. It should be
iterable (in the sense "stl iterable"). An example implementation of such a view class can be found in [Src/Utils/SpArrayView.hpp](./Src/Utils/SpArrayView.hpp). 

### Wrapper objects for callables
Callable wrappers select a backend: `SpCpu`, `SpCuda`, and `SpHip`. CUDA and HIP wrappers require the corresponding backend to be enabled. For example:

- SpCpu(\<CallableTy\> c)

Specifies that the callable c represents CPU code.

- SpCuda(\<CallableTy\> c)

Specifies that the callable c represents CUDA code.

- SpHip(\<CallableTy\> c)

Specifies that the callable c represents HIP code.

In all cases the callable c can be a lambda or a functor lvalue or rvalue.

A callable that appears as an argument of a call to the `task` method of a `SpRuntime` object without being wrapped into one the above two objects will be
interpreted by the runtime as CPU code by default. 

## Task viewer interface
The task viewer interface specifies ways to interact with the task object returned by the task method of SpRuntime.  
The exact type returned by the task method of SpRuntime is not important and in practice it should be type deduced (auto) in your programs.
You can however find the definition of the returned type in [Src/Task/SpAbstractTask.hpp](./Src/Task/SpAbstractTask.hpp).

### Main methods available on task objects returned by task calls
- bool isOver()

Returns true if the task has finished executing.

- void wait()

This method is a blocking call which waits until the task is finished.

- \<ReturnType\> getValue() 

This method is a blocking call which retrieves the result value of the task (if it has any).  
It first waits for the task to finish and then retrieves the result value.

- void setTaskName(const std::string& inTaskName)

Assigns the name inTaskName to the task.  
This change will be reflected in debug printouts, task graph and trace generation output.
By default the task will be named as the demangled string of the typeid name of the callable of the task. 

- std::string getTaskName()

Retrieves the name of the task.  
Speculative versions of tasks will have an apostrophe appended to their name.

# GPU/CUDA (Work-in-progress)

The CMake variable `SPECX_COMPILE_WITH_CUDA` must be set to ON, for example with the command `cmake .. -DSPECX_COMPILE_WITH_CUDA=ON`.
If CMake is not able to find nvcc, one must set `CUDACXX` env variable or the CMake variable `CMAKE_CUDA_COMPILER` to the path to nvcc.
On can set `CMAKE_CUDA_ARCHITECTURES` to select the CUDA sm to compile for.

Here is an example of a task on CUDA GPU:
```cpp
tg.task(SpWrite(a),// Dependencies are expressed as usually
    SpCuda([](SpDeviceDataView<std::vector<int>> paramA) { // Each parameter is converted into an SpDeviceDataView
        // The kernel call be called using the dedicated stream
        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.array(),
                                                           paramA.nbElements());
    })
);
```
Currently, the call to a CUDA kernel must be done in a `.cu` file.
There are three types of `SpDeviceDataView` that offer different methods: one for the `is_trivially_copyable` objects, one for the `std::vectors` of `is_trivially_copyable` objects, and one customize by the users.
In the latest, it is required to provide the following methods:
```cpp
    std::size_t memmovNeededSize() const{
    ...
    }

    template <class DeviceMemmov>
    void memmovHostToDevice(DeviceMemmov& mover, void* devicePtr, std::size_t size){
    ...
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr, std::size_t size){
    ...
    }
    
    auto getDeviceDataDescription() const{
    ...
    }
```
The type returned by `getDeviceDataDescription` must be copiable and have an empty constructor.
It should be used to help retreive the data from the raw pointer when calling a device kernel.

# GPU/HIP (Work-in-progress)

The CMake variable `SPECX_COMPILE_WITH_HIP` must be set to ON, for example with the command `cmake .. -DSPECX_COMPILE_WITH_HIP=ON`.
The C++ compiler must also be set with for example `CXX=hipcc`, so a working command line should be `CXX=hipcc cmake .. -DSPECX_COMPILE_WITH_HIP=ON`.
On can set `GPU_TARGETS` to select the HIP sm to compile for.
This has been tested with HIP version: 6.1.

Here is an example of a task on HIP GPU:
```cpp
tg.task(SpWrite(a),// Dependencies are expressed as usually
    SpHip([](SpDeviceDataView<std::vector<int>> paramA) { // Each parameter is converted into an SpDeviceDataView
        // The kernel call be called using the dedicated stream
        inc_var<<<1,1,0,SpHipUtils::GetCurrentStream()>>>(paramA.array(),
                                                           paramA.nbElements());
    })
);
```
Currently, the call to a HIP kernel must be done in a `.cu` file.
There are three types of `SpDeviceDataView` that offer different methods: one for the `is_trivially_copyable` objects, one for the `std::vectors` of `is_trivially_copyable` objects, and one customize by the users.
In the latest, it is required to provide the following methods:
```cpp
    std::size_t memmovNeededSize() const{
    ...
    }

    template <class DeviceMemmov>
    void memmovHostToDevice(DeviceMemmov& mover, void* devicePtr, std::size_t size){
    ...
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr, std::size_t size){
    ...
    }
    
    auto getDeviceDataDescription() const{
    ...
    }
```
The type returned by `getDeviceDataDescription` must be copiable and have an empty constructor.
It should be used to help retreive the data from the raw pointer when calling a device kernel.


# MPI

The CMake variable `SPECX_COMPILE_WITH_MPI` must be set to ON, for example with the command `cmake .. -DSPECX_COMPILE_WITH_MPI=ON`.

## Data Serialization and Deserialization
Data can be send to target MPI processes using the `mpiSend` and `mpiRecv` methods of the `SpTaskGraph` object.

To be moved across computing nodes, the objects must be one of the following types:
1. Be an instance of a class that inherits `SpAbstractSerializable` (see below for more details).
2. Supports the methods `getRawDataSize`, `getRawData` and `restoreRawData`, which will be used to extract the data to be sent, and restore them.
3. Be a POD type (well, have `is_standard_layout_v` and `is_trivial_v` returning true, which means that having a pointer in a struct will not be detected and could be an issue).
4. Be a vector of the types defined in 1, 2 or 3.

It is the function `SpGetSerializationType` that performs the detection and assigns the corresponding `SpSerializationType` value to each object.
The detection is performed in the order written above.

For examples, refer to the unit tests under `UTests/MPI`.

### Type 3 - POD
For built-in and POD types these methods work automatically:
```cpp
SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
int a = 1;
int b = 0;
...
tg.mpiSend(b, 1, 0);
tg.mpiRecv(b, 1, 1);
```

### Type 1 - SpAbstractSerializable
However, user-defined types must enable support for MPI serialization and deserialization. To do that they must implement these steps.
1. Include "MPI/SpSerializer.hpp"
2. Make the class a public subclass of the SpAbstractSerializable class
3. Provide a constructor that takes as argument a non-const referene to SpDeserializer. This constructor allows an object of the class to be constructed from the deserialization.
4. Provide a public method "serialize" with argument a non-const reference to SpSerializer. This method serializes the object into the input SpSerializer object.

These steps in detail are shown in the following example:
```cpp
#include "MPI/SpSerializer.hpp"

class int_data_holder : public SpAbstractSerializable {
public:
	int_data_holder(int value = 0) : value{value} {}
	int_data_holder(SpDeserializer &deserializer) : value(deserializer.restore<decltype(value)>("value")) {
	}
	
	void serialize(SpSerializer &serializer) const final {
		serializer.append(value, "value");
	}

	int get() const { return value; }
	void set(int value) {
		this->value = value;
	}
private:
	int value;
};
...
SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
int_data_holder a = 1;
int_data_holder b = 0;
...
tg.mpiSend(b, 1, 0);
tg.mpiRecv(b, 1, 1);
```

### Type 2 - Direct access

```cpp
class DirectAccessClass {
    int key;
public:
    const unsigned char* getRawData() const {
        return reinterpret_cast<const unsigned char*>(&key);
    }
    std::size_t getRawDataSize() const {
        return sizeof(key);
    }

    void restoreRawData(const unsigned char* ptr, std::size_t size){
        assert(sizeof(key) == size);
        key = *reinterpret_cast<const int*>(ptr);
    }

    int& value(){
        return key;
    }

    const int& value() const{
        return key;
    }
};
```

