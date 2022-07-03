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

    //////////////////////////////////////////////////////////////////

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
            memset(request, 0, sizeof(request));
            memset(bufferSize, 0, sizeof(bufferSize));
        }

        virtual ~SpMpiSendTransaction(){
            DpAssertMpi(MPI_Request_free(&request));
            DpAssertMpi(MPI_Request_free(&requestBufferSize));
        }
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

        SpMpiRecvTransaction(){
            task = nullptr;
            tm = nullptr;
            atg = nullptr;
            memset(request, 0, sizeof(request));
            memset(bufferSize, 0, sizeof(bufferSize));
        }

        virtual ~SpMpiRecvTransaction(){
            DpAssertMpi(MPI_Request_free(&request));
            DpAssertMpi(MPI_Request_free(&requestBufferSize));
        }
    };

    //////////////////////////////////////////////////////////////////

    struct SpRequestType{
        bool isSend = true;
        int state = 0;
        int idxTransaction = -1;
    };

    static void consume(SpMpiBackgroundWorker* data) {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(data->queueMutex);
                data->mutexCondition.wait(lock, [data] {
                    return !data->newSends.empty()
                            || !data->newRecvs.empty()
                            || data->shouldTerminate;
                });
                if (data->shouldTerminate) {
                    assert(data->newSends.empty()
                           && data->newRecvs.empty()
                           && data->sendTransactions.empty()
                           && data->recvTransactions.empty());
                    return;
                }
                while(!data->newSends.empty()){
                    sendTransactions[counterTransactions++] = (data->newSends.back()());
                    data->newSends.pop_back();

                    SpMpiSendTransaction& tr = sendTransactions.back();
                    allRequestsTypes.emplace_back(SpRequestType{true, 0, int(sendTransactions.size()-1)});
                    allRequests.emplace_back(tr.requestBufferSize);
                    allRequestsTypes.emplace_back(SpRequestType{true, 1, int(sendTransactions.size()-1)});
                    allRequests.emplace_back(tr.request);
                }
                while(!data->newRecvs.empty()){
                    recvTransactions[counterTransactions++] = (data->newRecvs.back()());
                    data->newRecvs.pop_back();

                    SpMpiSendTransaction& tr = recvTransactions.back();
                    allRequestsTypes.emplace_back(SpRequestType{false, 0, int(sendTransactions.size()-1)});
                    allRequests.emplace_back(tr.requestBufferSize);
                }
            }
            int idxDone = MPI_UNDEFINED;
            do{
                SpAssertMpi(MPI_Testany(static_cast<int>(allRequests.size()), allRequests.data(), &idxDone, MPI_STATUSES_IGNORE));
                if(idxDone != MPI_UNDEFINED){
                    SpRequestType rt = allRequestsTypes[idxDone];
                    std::swap(allRequestsTypes[idxDone], allRequestsTypes.back());
                    allRequestsTypes.push_back();
                    std::swap(allRequests[idxDone], allRequests.back());
                    allRequests.push_back();

                    if(rt.isSend){
                        if(rt.state == 1){
                            // Send done
                            SpMpiRecvTransaction transaction = std::move(sendTransactions[rt.idxTransaction]);
                            sendTransactions.erase(rt.idxTransaction);
                            // Post back task
                            transaction.tm->postMPITaskExecution(*transaction.atg, transaction.task);
                        }
                    }
                    else{
                        if(rt.state == 0){
                            // Size recv
                            SpMpiRecvTransaction& transaction = recvTransactions[rt.idxTransaction];
                            transaction.buffer.resize(transaction.buffer.size());
                            transaction.request = DpIrecv(transaction.buffer.data(),
                                                          int(transaction.buffer.size()),
                                                          srcProc, tag, mpiCom);
                            allRequestsTypes.emplace_back(SpRequestType{false, 1, rt.idxTransaction});
                            allRequests.emplace_back(tr.request);
                        }
                        else if(rt.state == 1){
                            // Recv done
                            SpMpiRecvTransaction transaction = std::move(recvTransactions[rt.idxTransaction]);
                            recvTransactions.erase(rt.idxTransaction);
                            transaction.deserializer(transaction.buffer.data(),
                                                     int(transaction.buffer.size()));
                            // Post back task
                            transaction.tm->postMPITaskExecution(*transaction.atg, transaction.task);
                        }
                    }
                }
            } while(idxDone != MPI_UNDEFINED && allRequests.size());
        }
    }


    bool shouldTerminate = false;
    std::mutex queueMutex;
    std::condition_variable mutexCondition;
    std::vector<std::function<SpMpiSendTransaction()>> newSends;
    std::vector<std::function<SpMpiRecvTransaction()>> newRecvs;

    int counterTransactions;
    std::unordered_map<int, SpMpiSendTransaction> sendTransactions;
    std::unordered_map<int, SpMpiRecvTransaction> recvTransactions;

    std::thread thread;

    MPI_Comm mpiCom;
    std::vector<MPI_Request> allRequests;
    std::vecotr<SpRequestType> allRequestsTypes;

public:
    SpMpiBackgroundWorker()
        : counterTransactions(0), thread(consume, this){

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
            newRecvs.push_back(std::move(com));
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
