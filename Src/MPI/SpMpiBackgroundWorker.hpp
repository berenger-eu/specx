#ifndef SPMPIBACKGROUNDWORKER_HPP
#define SPMPIBACKGROUNDWORKER_HPP

#include "Config/SpConfig.hpp"

#ifndef SPETABARU_COMPILE_WITH_MPI
#error MPI but be enable to use this file.
#endif

#include <thread>
#include <functional>
#include <future>
#include <queue>

#include <mpi.h>

class SpMpiBackgroundWorker {

    template <class ObjectType>
    static MPI_Request DpIsend(const ObjectType data[], const int nbElements, const int dest, const int tag, const MPI_Comm inCom){
        MPI_Request request;
        DpAssertMpi(MPI_Isend(const_cast<ObjectType*>(data), nbElements, DpGetMpiType<ObjectType>::type, dest,
                              tag,
                              inCom, &request));
        return request;
    }

    template <class ObjectType>
    static MPI_Request DpIrecv(ObjectType data[], const int nbElements, const int dest, const int tag, const MPI_Comm inCom){
        MPI_Request request;
        DpAssertMpi(MPI_Irecv(data, nbElements, DpGetMpiType<ObjectType>::type, dest,
                              tag,
                              inCom, &request));
        return request;
    }

    std::vector<MPI_Request> allRequests;

    struct  {
        SpAbstractTask* task;
    };

    SpAssertMpi(MPI_Testany(static_cast<int>(allRequests.size()), allRequests.data(), &idxDone, MPI_STATUSES_IGNORE));

    static void consume(SpMpiBackgroundWorker* data) {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(data->queueMutex);
                data->mutexCondition.wait(lock, [data] {
                    return !data->queueJobs.empty() || data->shouldTerminate;
                });
                if (data->queueJobs.empty() && data->shouldTerminate) {
                    return;
                }
                job = data->queueJobs.front();
                data->queueJobs.pop();
            }
            job();
        }
    }


    bool shouldTerminate = false;
    std::mutex queueMutex;
    std::condition_variable mutexCondition;
    std::queue<std::function<void()>> queueJobs;
    std::thread thread;

    MPI_Comm mpiCom;

public:
    SpMpiBackgroundWorker()
        : thread(consume, this){

    }

    SpMpiBackgroundWorker(const SpMpiBackgroundWorker&) = delete;
    SpMpiBackgroundWorker(SpMpiBackgroundWorker&&) = delete;
    SpMpiBackgroundWorker& operator=(const SpMpiBackgroundWorker&) = delete;
    SpMpiBackgroundWorker& operator=(SpMpiBackgroundWorker&&) = delete;

    ~SpMpiBackgroundWorker(){
        if(shouldTerminate == false){
            stop();
        }
    }

    template <class Serializer>
    void addSend(Serializer& serializer, SpDataHandle* handle,
                    const int destProc, const int tag) {
        auto comJob = [=,serializer]() -> MPI_Request{
            std::vector<unsigned char> data = serializer(handle);
            return DpIsend(data.data(), int(data.size()), destProc, tag, mpiCom);
        };
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueJobs.push(std::move(comJob));
        }
        mutexCondition.notify_one();
    }

    template <class Deserializer>
    void addRecv(Deserializer& deserializer, SpDataHandle* handle,
                    const int srcProc, const int tag) {
        auto comJob = [=,deserializer]() -> MPI_Request{
            std::vector<unsigned char> data = serializer(handle);
            return DpIsend(data.data(), int(data.size()), destProc, tag, mpiCom);
        };
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueJobs.push(std::move(com));
        }
        mutexCondition.notify_one();
    }

    void stop() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            shouldTerminate = true;
        }
        mutexCondition.notify_all();
        thread.join();
    }
};


#endif // SPMPIBACKGROUNDWORKER_HPP
