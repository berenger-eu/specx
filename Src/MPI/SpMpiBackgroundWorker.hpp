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
        if(SpDebug::Controller.isEnable()){
            SpDebugPrint() << "[SpMpiUtils::Isend] => nbElements " << nbElements << " dest " << dest
                       << " tag " << tag;
        }
        MPI_Request request;
        SpAssertMpi(MPI_Isend(const_cast<ObjectType*>(data), nbElements, SpGetMpiType<ObjectType>(), dest,
                              tag,
                              inCom, &request));
        return request;
    }

    /// Interface to mpi irecv
    template <class ObjectType>
    static MPI_Request Irecv(ObjectType data[], const int nbElements, const int src, const int tag, const MPI_Comm inCom){
        if(SpDebug::Controller.isEnable()){
            SpDebugPrint() << "[SpMpiUtils::Irecv] => nbElements " << nbElements << " src " << src
                       << " tag " << tag;
        }
        MPI_Request request;
        SpAssertMpi(MPI_Irecv(data, nbElements, SpGetMpiType<ObjectType>(), src,
                              tag,
                              inCom, &request));
        return request;
    }

    /// Interface to mpi isend
    template <class ObjectType>
    static MPI_Request IBroadcastsend(const ObjectType data[], const int nbElements, const int root, const MPI_Comm inCom){
        if(SpDebug::Controller.isEnable()){
            SpDebugPrint() << "[SpMpiUtils::IBroadcastsend] => nbElements " << nbElements << " root " << root
                           << " root " << root;
        }
        MPI_Request request;
        SpAssertMpi(MPI_Ibcast(const_cast<ObjectType*>(data), nbElements, SpGetMpiType<ObjectType>(), root,
                              inCom, &request));
        return request;
    }

    /// Interface to mpi irecv
    template <class ObjectType>
    static MPI_Request IBroadcastrecv(ObjectType data[], const int nbElements, const int root, const MPI_Comm inCom){
        if(SpDebug::Controller.isEnable()){
            SpDebugPrint() << "[SpMpiUtils::IBroadcastrecv] => nbElements " << nbElements << " root " << root
                           << " root " << root;
        }
        MPI_Request request;
        SpAssertMpi(MPI_Ibcast(data, nbElements, SpGetMpiType<ObjectType>(), root,
                              inCom, &request));
        return request;
    }


    //////////////////////////////////////////////////////////////////

    enum SpRequestType{
        TYPE_SEND,
        TYPE_RECV,
        TYPE_BROADCASTSEND,
        TYPE_BROADCASTRECV,
        TYPE_UNDEFINED
    };

    enum SpRequestState{
        STATE_FIRST,
        STATE_SECOND,
        STATE_UNDEFINED
    };

    struct SpRequest{
        SpRequestType type = TYPE_UNDEFINED;
        SpRequestState state = STATE_UNDEFINED;
        int idxTransaction = -1;
    };

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



    struct SpMpiBroadcastSendRecvTransaction {
        SpTaskManager* tm;
        SpAbstractTaskGraph* atg;

        SpAbstractTask* task;

        MPI_Request requestBufferSize;
        std::unique_ptr<int> bufferSize;

        MPI_Request request;
        std::unique_ptr<SpAbstractMpiSerializer> serializer;

        int root;

        std::vector<unsigned char> buffer;
        std::unique_ptr<SpAbstractMpiDeSerializer> deserializer;

        SpRequestType type;

        int broadcastTicket;

        SpMpiBroadcastSendRecvTransaction(){
            task = nullptr;
            tm = nullptr;
            atg = nullptr;
            memset(&request, 0, sizeof(request));
            memset(&requestBufferSize, 0, sizeof(requestBufferSize));
            root = -1;
            type = TYPE_UNDEFINED;
            broadcastTicket = -1;
        }

        SpMpiBroadcastSendRecvTransaction(const SpMpiBroadcastSendRecvTransaction&) = delete;
        SpMpiBroadcastSendRecvTransaction(SpMpiBroadcastSendRecvTransaction&&) = default;
        SpMpiBroadcastSendRecvTransaction& operator=(const SpMpiBroadcastSendRecvTransaction&) = delete;
        SpMpiBroadcastSendRecvTransaction& operator=(SpMpiBroadcastSendRecvTransaction&&) = default;

        void releaseRequest(){
            // TODO, it seems that we should not free non consistant req.
            // SpAssertMpi(MPI_Request_free(&request));
            // SpAssertMpi(MPI_Request_free(&requestBufferSize));
        }
    };


    struct BroadcastOrder{
        bool operator()(const std::pair<int, std::function<SpMpiBroadcastSendRecvTransaction()>>& lhs,
                        const std::pair<int, std::function<SpMpiBroadcastSendRecvTransaction()>>& rhs){
            return lhs.first > rhs.first;
        }
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
    std::priority_queue<std::pair<int, std::function<SpMpiBroadcastSendRecvTransaction()>>,
                    std::vector<std::pair<int, std::function<SpMpiBroadcastSendRecvTransaction()>>>,
                    BroadcastOrder> newBroadcastSendRecvs;

    const bool isInit;
    MPI_Comm mpiCom;
    int broadcastCpt;
    std::thread thread;

    SpMpiBackgroundWorker()
        : isInit(Init()), mpiCom(MPI_COMM_WORLD), broadcastCpt(0), thread(Consume, this){

    }
    static SpMpiBackgroundWorker MainWorker;

public:
    static constexpr int OffsetTagSize = 9999;

    SpMpiBackgroundWorker(const SpMpiBackgroundWorker&) = delete;
    SpMpiBackgroundWorker(SpMpiBackgroundWorker&&) = delete;
    SpMpiBackgroundWorker& operator=(const SpMpiBackgroundWorker&) = delete;
    SpMpiBackgroundWorker& operator=(SpMpiBackgroundWorker&&) = delete;

    static SpMpiBackgroundWorker& GetWorker(){
        return MainWorker;
    }

    void init(){}

    ~SpMpiBackgroundWorker(){
        if(SpDebug::Controller.isEnable()){
            SpDebugPrint() << "[SpMpiBackgroundWorker] => ~SpMpiBackgroundWorker ";
        }
        if(shouldTerminate == false){
            stop();
        }
        SpAssertMpi(MPI_Finalize());
        if(SpDebug::Controller.isEnable()){
            SpDebugPrint() << "[SpMpiBackgroundWorker] => MPI_Finalize done ";
        }
    }

    int getAndIncBroadcastCpt(){
        return broadcastCpt++;
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

            transaction.serializer.reset(new SpMpiSerializer<SpGetSerializationType<ObjectType>(), ObjectType>(obj));

            transaction.bufferSize.reset(new int(transaction.serializer->getBufferSize()));
            transaction.requestBufferSize = Isend(transaction.bufferSize.get(),
                                            1, destProc, tag+OffsetTagSize, mpiCom);

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

            transaction.deserializer.reset(new SpMpiDeSerializer<SpGetSerializationType<ObjectType>(), ObjectType>(obj));

            transaction.bufferSize.reset(new int(0));
            transaction.requestBufferSize = Irecv(transaction.bufferSize.get(),
                                          1, srcProc, tag+OffsetTagSize, mpiCom);
            return transaction;
        };
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            newRecvs.push_back(std::move(comJob));
        }
        mutexCondition.notify_one();
    }


    template <class ObjectType>
    void addBroadcastSend(const ObjectType& obj, const int root, const int broadcastTicket,
                 SpAbstractTask* task,
                 SpTaskManager* tm,
                 SpAbstractTaskGraph* atg) {
        auto comJob = [=, &obj]() -> SpMpiBroadcastSendRecvTransaction {
            SpMpiBroadcastSendRecvTransaction transaction;
            transaction.type = TYPE_BROADCASTSEND;
            transaction.task = task;
            transaction.tm = tm;
            transaction.atg = atg;
            transaction.broadcastTicket = broadcastTicket;

            transaction.serializer.reset(new SpMpiSerializer<SpGetSerializationType<ObjectType>(), ObjectType>(obj));

            transaction.bufferSize.reset(new int(transaction.serializer->getBufferSize()));
            transaction.requestBufferSize = IBroadcastsend(transaction.bufferSize.get(),
                                            1, root, mpiCom);

            transaction.request = IBroadcastsend(transaction.serializer->getBuffer(),
                                          transaction.serializer->getBufferSize(),
                                          root, mpiCom);
            return transaction;
        };
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            newBroadcastSendRecvs.push({ broadcastTicket, std::move(comJob)});
        }
        mutexCondition.notify_one();
    }

    template <class ObjectType>
    void addBroadcastRecv(ObjectType& obj, const int root, const int broadcastTicket,
                 SpAbstractTask* task,
                 SpTaskManager* tm,
                 SpAbstractTaskGraph* atg) {
        auto comJob = [=, &obj]() -> SpMpiBroadcastSendRecvTransaction{
            SpMpiBroadcastSendRecvTransaction transaction;
            transaction.type = TYPE_BROADCASTRECV;
            transaction.task = task;
            transaction.tm = tm;
            transaction.atg = atg;
            transaction.root = root;
            transaction.broadcastTicket = broadcastTicket;

            transaction.deserializer.reset(new SpMpiDeSerializer<SpGetSerializationType<ObjectType>(), ObjectType>(obj));

            transaction.bufferSize.reset(new int(0));
            transaction.requestBufferSize = IBroadcastrecv(transaction.bufferSize.get(),
                                          1, root, mpiCom);
            return transaction;
        };
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            newBroadcastSendRecvs.push({ broadcastTicket, std::move(comJob)});
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
