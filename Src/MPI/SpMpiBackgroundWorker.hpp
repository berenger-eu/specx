#ifndef SPMPIBACKGROUNDWORKER_HPP
#define SPMPIBACKGROUNDWORKER_HPP

#include "Config/SpConfig.hpp"

#ifndef SPECX_COMPILE_WITH_MPI
#error MPI but be enable to use this file.
#endif

#include "SpMpiUtils.hpp"
#include "SpMpiTypeUtils.hpp"
#include "SpMpiSerializer.hpp"
#include "Utils/SpDebug.hpp"

#include <thread>
#include <functional>
#include <future>
#include <queue>

#include <mpi.h>


class SpTaskManager;
class SpAbstractTaskGraph;
class SpAbstractTask;

///
/// \brief The SpMpiBackgroundWorker class is a singleton
/// with a thread running that will mamange all the communications.
/// The communications are pushed into the singleton, then
/// a test any is perform with a spin loop.
///
class SpMpiBackgroundWorker {

    //////////////////////////////////////////////////////////////////

    /// Interface to mpi isend
    template <class ObjectType>
    static MPI_Request Isend(const ObjectType data[], const int nbElements, const int dest, const int tag, const MPI_Comm inCom){
        SpDebugPrint() << "[SpMpiUtils::Isend] => nbElements " << nbElements << " dest " << dest
                       << " tag " << tag;

        MPI_Request request;
        SpAssertMpi(MPI_Isend(const_cast<ObjectType*>(data), nbElements, SpGetMpiType<ObjectType>::type, dest,
                              tag,
                              inCom, &request));
        return request;
    }

    /// Interface to mpi irecv
    template <class ObjectType>
    static MPI_Request Irecv(ObjectType data[], const int nbElements, const int src, const int tag, const MPI_Comm inCom){
        SpDebugPrint() << "[SpMpiUtils::Irecv] => nbElements " << nbElements << " src " << src
                       << " tag " << tag;

        MPI_Request request;
        SpAssertMpi(MPI_Irecv(data, nbElements, SpGetMpiType<ObjectType>::type, src,
                              tag,
                              inCom, &request));
        return request;
    }

    //////////////////////////////////////////////////////////////////


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
            memset(&requestBufferSize, 0, sizeof(requestBufferSize));
        }

        SpMpiSendTransaction(const SpMpiSendTransaction&) = delete;
        SpMpiSendTransaction(SpMpiSendTransaction&&) = default;
        SpMpiSendTransaction& operator=(const SpMpiSendTransaction&) = delete;
        SpMpiSendTransaction& operator=(SpMpiSendTransaction&&) = default;

        void releaseRequest(){
            // TODO, it seems that we should not free non consistant req.
            // SpAssertMpi(MPI_Request_free(&request));
            // SpAssertMpi(MPI_Request_free(&requestBufferSize));
        }
    };

    //////////////////////////////////////////////////////////////////


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
            memset(&requestBufferSize, 0, sizeof(requestBufferSize));
            srcProc = 0;
            tag = 0;
        }

        SpMpiRecvTransaction(const SpMpiRecvTransaction&) = delete;
        SpMpiRecvTransaction(SpMpiRecvTransaction&&) = default;
        SpMpiRecvTransaction& operator=(const SpMpiRecvTransaction&) = delete;
        SpMpiRecvTransaction& operator=(SpMpiRecvTransaction&&) = default;

        void releaseRequest(){
            // TODO, it seems that we should not free non consistant req.
            // SpAssertMpi(MPI_Request_free(&request));
            // SpAssertMpi(MPI_Request_free(&requestBufferSize));
        }
    };

    //////////////////////////////////////////////////////////////////

    struct SpRequestType{
        bool isSend = true;
        int state = 0;
        int idxTransaction = -1;
    };

    static void Consume(SpMpiBackgroundWorker* data);

    static bool Init(){
        SpAssertMpi(MPI_Init(nullptr, nullptr));
        return true;
    }

    bool shouldTerminate = false;
    std::mutex queueMutex;
    std::condition_variable mutexCondition;
    std::vector<std::function<SpMpiSendTransaction()>> newSends;
    std::vector<std::function<SpMpiRecvTransaction()>> newRecvs;

    const bool isInit;
    MPI_Comm mpiCom;

    std::thread thread;

    SpMpiBackgroundWorker()
        : isInit(Init()), mpiCom(MPI_COMM_WORLD), thread(Consume, this){

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
        SpAssertMpi(MPI_Finalize());
    }

    template <class ObjectType>
    void addSend(const ObjectType& obj, const int destProc, const int tag,
                 SpAbstractTask* task,
                 SpTaskManager* tm,
                 SpAbstractTaskGraph* atg) {
        auto comJob = [=, &obj]() -> SpMpiSendTransaction {
            SpMpiSendTransaction transaction;
            transaction.task = task;
            transaction.tm = tm;
            transaction.atg = atg;

            transaction.serializer.reset(new SpMpiSerializer<ObjectType>(obj));

            transaction.bufferSize.reset(new int(transaction.serializer->getBufferSize()));
            transaction.requestBufferSize = Isend(transaction.bufferSize.get(),
                                            1, destProc, tag+9999, mpiCom);

            transaction.request = Isend(transaction.serializer->getBuffer(),
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

    template <class ObjectType>
    void addRecv(ObjectType& obj, const int srcProc, const int tag,
                 SpAbstractTask* task,
                 SpTaskManager* tm,
                 SpAbstractTaskGraph* atg) {
        auto comJob = [=, &obj]() -> SpMpiRecvTransaction{
            SpMpiRecvTransaction transaction;
            transaction.task = task;
            transaction.tm = tm;
            transaction.atg = atg;
            transaction.srcProc = srcProc;
            transaction.tag = tag;

            transaction.deserializer.reset(new SpMpiDeSerializer<ObjectType>(obj));

            transaction.bufferSize.reset(new int(0));
            transaction.requestBufferSize = Irecv(transaction.bufferSize.get(),
                                          1, srcProc, tag+9999, mpiCom);
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
