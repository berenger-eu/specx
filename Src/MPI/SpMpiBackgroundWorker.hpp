#ifndef SPMPIBACKGROUNDWORKER_HPP
#define SPMPIBACKGROUNDWORKER_HPP

#include "Config/SpConfig.hpp"

#ifndef SPETABARU_COMPILE_WITH_MPI
#error MPI but be enable to use this file.
#endif

#include "SpMPIUtils.hpp"
#include "Task/SpAbstractTask.hpp"

#include <thread>
#include <functional>
#include <future>
#include <queue>

#include <mpi.h>


class SpTaskManager;
class SpAbstractTaskGraph;

class SpMpiBackgroundWorker {

    //////////////////////////////////////////////////////////////////

    template <class ObjectType>
    static MPI_Request DpIsend(const ObjectType data[], const int nbElements, const int dest, const int tag, const MPI_Comm inCom){
        MPI_Request request;
        SpAssertMpi(MPI_Isend(const_cast<ObjectType*>(data), nbElements, DpGetMpiType<ObjectType>::type, dest,
                              tag,
                              inCom, &request));
        return request;
    }

    template <class ObjectType>
    static MPI_Request DpIrecv(ObjectType data[], const int nbElements, const int dest, const int tag, const MPI_Comm inCom){
        MPI_Request request;
        SpAssertMpi(MPI_Irecv(data, nbElements, DpGetMpiType<ObjectType>::type, dest,
                              tag,
                              inCom, &request));
        return request;
    }

    //////////////////////////////////////////////////////////////////

    class SpAbstractMpiSerializer {
    public:
        virtual ~SpAbstractMpiSerializer(){}
        virtual unsigned char* getBuffer() = 0;
        virtual int getBufferSize() = 0;
    };

    template <class SerializerClass>
    struct SpMpiSerializer : public SpAbstractMpiSerializer {
        SerializerClass serializer;

        virtual unsigned char* getBuffer() override{
            return serializer.getBuffer();
        }
        virtual int getBufferSize() override{
            return serializer.getBufferSize();
        }
    };

    struct SpMpiSendTransaction {
        SpTaskManager* tm;
        SpAbstractTaskGraph* atg;

        SpAbstractTask* task;

        MPI_Request requestBufferSize;
        std::unique_ptr<int> bufferSize;

        MPI_Request request;
        std::unique_ptr<SpAbstractMpiSerializer> serializer;

        SpMpiSendTransaction(){
            task = nullptr;
            tm = nullptr;
            atg = nullptr;
            memset(&request, 0, sizeof(request));
        }

        virtual ~SpMpiSendTransaction(){
            SpAssertMpi(MPI_Request_free(&request));
            SpAssertMpi(MPI_Request_free(&requestBufferSize));
        }

        SpMpiSendTransaction(const SpMpiSendTransaction&) = delete;
        SpMpiSendTransaction(SpMpiSendTransaction&&) = default;
        SpMpiSendTransaction& operator=(const SpMpiSendTransaction&) = delete;
        SpMpiSendTransaction& operator=(SpMpiSendTransaction&&) = default;
    };

    //////////////////////////////////////////////////////////////////

    class SpAbstractMpiDeSerializer {
    public:
        virtual ~SpAbstractMpiDeSerializer(){}
        virtual void deserialize(unsigned char* buffer, int bufferSize) = 0;
    };

    template <class DeserializerClass>
    struct SpMpiDeSerializer : public SpAbstractMpiDeSerializer {
        DeserializerClass deserializer;

        void deserialize(unsigned char* buffer, int bufferSize) override{
            deserializer.deserialize(buffer, bufferSize);
        }
    };

    struct SpMpiRecvTransaction {
        SpTaskManager* tm;
        SpAbstractTaskGraph* atg;

        MPI_Request request;
        SpAbstractTask* task;
        MPI_Request requestBufferSize;
        std::unique_ptr<int> bufferSize;
        std::vector<unsigned char> buffer;
        std::unique_ptr<SpAbstractMpiDeSerializer> deserializer;

        int srcProc;
        int tag;

        SpMpiRecvTransaction(){
            task = nullptr;
            tm = nullptr;
            atg = nullptr;
            memset(&request, 0, sizeof(request));
            srcProc = 0;
            tag = 0;
        }

        virtual ~SpMpiRecvTransaction(){
            SpAssertMpi(MPI_Request_free(&request));
            SpAssertMpi(MPI_Request_free(&requestBufferSize));
        }

        SpMpiRecvTransaction(const SpMpiRecvTransaction&) = delete;
        SpMpiRecvTransaction(SpMpiRecvTransaction&&) = default;
        SpMpiRecvTransaction& operator=(const SpMpiRecvTransaction&) = delete;
        SpMpiRecvTransaction& operator=(SpMpiRecvTransaction&&) = default;
    };

    //////////////////////////////////////////////////////////////////

    struct SpRequestType{
        bool isSend = true;
        int state = 0;
        int idxTransaction = -1;
    };

    static void Consume(SpMpiBackgroundWorker* data);


    bool shouldTerminate = false;
    std::mutex queueMutex;
    std::condition_variable mutexCondition;
    std::vector<std::function<SpMpiSendTransaction()>> newSends;
    std::vector<std::function<SpMpiRecvTransaction()>> newRecvs;

    MPI_Comm mpiCom;

    std::thread thread;

    SpMpiBackgroundWorker()
        : mpiCom(MPI_COMM_WORLD), thread(Consume, this){

    }
    static SpMpiBackgroundWorker MainWorker;

public:
    SpMpiBackgroundWorker(const SpMpiBackgroundWorker&) = delete;
    SpMpiBackgroundWorker(SpMpiBackgroundWorker&&) = delete;
    SpMpiBackgroundWorker& operator=(const SpMpiBackgroundWorker&) = delete;
    SpMpiBackgroundWorker& operator=(SpMpiBackgroundWorker&&) = delete;

    static SpMpiBackgroundWorker& GetWorker(){
        return MainWorker;
    }

    ~SpMpiBackgroundWorker(){
        if(shouldTerminate == false){
            stop();
        }
    }

    template <class Serializer>
    void addSend(const int destProc, const int tag,
                 SpAbstractTask* task,
                 SpTaskManager* tm,
                 SpAbstractTaskGraph* atg) {
        auto comJob = [=]() -> SpMpiSendTransaction {
            SpMpiSendTransaction transaction;
            transaction.task = task;
            transaction.tm = tm;
            transaction.atg = atg;

            auto handles = task->getDataHandles();
            assert(handles.size() == 1);
            SpDataHandle* handle = handles[0].first;
            transaction.serializer(new Serializer(handle));

            transaction.bufferSize(new int(transaction.serializer->getBufferSize()));
            transaction.requestBufferSize = DpIsend(transaction.bufferSize.get(),
                                            1, destProc, tag, mpiCom);

            transaction.request = DpIsend(transaction.serializer->getBuffer(),
                                          transaction.serializer->getBufferSize(),
                                          destProc, tag, mpiCom);
            return transaction;
        };
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            newSends.push_back(std::move(comJob));
        }
        mutexCondition.notify_one();
    }

    template <class Deserializer>
    void addRecv(const int srcProc, const int tag,
                 SpAbstractTask* task,
                 SpTaskManager* tm,
                 SpAbstractTaskGraph* atg) {
        auto comJob = [=]() -> SpMpiRecvTransaction{
            SpMpiRecvTransaction transaction;
            transaction.task = task;
            transaction.tm = tm;
            transaction.atg = atg;
            transaction.srcProc = srcProc;
            transaction.tag = tag;

            auto handles = task->getDataHandles();
            assert(handles.size() == 1);
            SpDataHandle* handle = handles[0].first;
            transaction.deserializer(new Deserializer(handle));

            transaction.requestBufferSize = DpIrecv(transaction.bufferSize.get(),
                                          1, srcProc, tag, mpiCom);
            return transaction;
        };
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            newRecvs.push_back(std::move(comJob));
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
