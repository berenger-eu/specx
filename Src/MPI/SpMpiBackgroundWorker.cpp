#include "SpMpiBackgroundWorker.hpp"
#include "Scheduler/SpTaskManager.hpp"


SpMpiBackgroundWorker SpMpiBackgroundWorker::MainWorker;


void SpMpiBackgroundWorker::Consume(SpMpiBackgroundWorker* data) {
    std::vector<MPI_Request> allRequests;
    std::vector<SpRequestType> allRequestsTypes;

    std::unordered_map<int, SpMpiSendTransaction> sendTransactions;
    std::unordered_map<int, SpMpiRecvTransaction> recvTransactions;

    int counterTransactions(0);

    while (true) {
        {
            std::unique_lock<std::mutex> lock(data->queueMutex);
            data->mutexCondition.wait(lock, [data, &allRequests] {
                return !data->newSends.empty()
                        || !data->newRecvs.empty()
                        || !allRequests.empty()
                        || data->shouldTerminate;
            });
            if (data->shouldTerminate) {
                assert(data->newSends.empty()
                       && data->newRecvs.empty()
                       && sendTransactions.empty()
                       && recvTransactions.empty()
                       && allRequests.empty());
                break;
            }
            while(!data->newSends.empty()){
                auto func = std::move(data->newSends.back());
                data->newSends.pop_back();
                sendTransactions[counterTransactions] = func();

                SpMpiSendTransaction& tr = sendTransactions[counterTransactions];
                allRequestsTypes.emplace_back(SpRequestType{true, 0, counterTransactions});
                allRequests.emplace_back(tr.requestBufferSize);
                allRequestsTypes.emplace_back(SpRequestType{true, 1, counterTransactions});
                allRequests.emplace_back(tr.request);
                counterTransactions += 1;
            }
            while(!data->newRecvs.empty()){
                auto func = data->newRecvs.back();
                data->newRecvs.pop_back();
                recvTransactions[counterTransactions] = func();

                SpMpiRecvTransaction& tr = recvTransactions[counterTransactions];
                allRequestsTypes.emplace_back(SpRequestType{false, 0, counterTransactions});
                allRequests.emplace_back(tr.requestBufferSize);
                counterTransactions += 1;
            }
        }
        int flagDone = 0;
        do{
            usleep(10000);
            int idxDone = MPI_UNDEFINED;
            SpAssertMpi(MPI_Testany(static_cast<int>(allRequests.size()), allRequests.data(), &idxDone, &flagDone, MPI_STATUS_IGNORE));
            if(flagDone){
                if(SpDebug::Controller.isEnable()){
                    SpDebugPrint() << "[SpMpiBackgroundWorker] => idxDone " << idxDone;
                }

                assert(idxDone != MPI_UNDEFINED);
                SpRequestType rt = allRequestsTypes[idxDone];
                std::swap(allRequestsTypes[idxDone], allRequestsTypes.back());
                allRequestsTypes.pop_back();
                std::swap(allRequests[idxDone], allRequests.back());
                allRequests.pop_back();

                if(rt.isSend){
                    assert(sendTransactions.find(rt.idxTransaction) != sendTransactions.end());
                    if(SpDebug::Controller.isEnable()){
                        SpDebugPrint() << "[SpMpiBackgroundWorker] => send done " << rt.idxTransaction;
                    }
                    if(rt.state == 1){
                        if(SpDebug::Controller.isEnable()){
                            SpDebugPrint() << "[SpMpiBackgroundWorker] => send complete " << rt.idxTransaction;
                        }
                        // Send done
                        SpMpiSendTransaction transaction = std::move(sendTransactions[rt.idxTransaction]);
                        sendTransactions.erase(rt.idxTransaction);
                        // Post back task
                        transaction.tm->postMPITaskExecution(*transaction.atg, transaction.task);
                        transaction.releaseRequest();
                    }
                }
                else{
                    assert(recvTransactions.find(rt.idxTransaction) != recvTransactions.end());
                    if(rt.state == 0){
                        if(SpDebug::Controller.isEnable()){
                            SpDebugPrint() << "[SpMpiBackgroundWorker] => recv state 0 " << rt.idxTransaction;
                        }
                        // Size recv
                        SpMpiRecvTransaction& transaction = recvTransactions[rt.idxTransaction];
                        transaction.buffer.resize(*transaction.bufferSize);
                        transaction.request = Irecv(transaction.buffer.data(),
                                                      int(transaction.buffer.size()),
                                                      transaction.srcProc, transaction.tag, data->mpiCom);
                        allRequestsTypes.emplace_back(SpRequestType{false, 1, rt.idxTransaction});
                        allRequests.emplace_back(transaction.request);
                    }
                    else if(rt.state == 1){
                        if(SpDebug::Controller.isEnable()){
                            SpDebugPrint() << "[SpMpiBackgroundWorker] => recv state 1 " << rt.idxTransaction;
                        }
                        // Recv done
                        SpMpiRecvTransaction transaction = std::move(recvTransactions[rt.idxTransaction]);
                        recvTransactions.erase(rt.idxTransaction);
                        transaction.deserializer->deserialize(transaction.buffer.data(),
                                                              int(transaction.buffer.size()));
                        // Post back task
                        transaction.tm->postMPITaskExecution(*transaction.atg, transaction.task);
                        transaction.releaseRequest();
                    }
                }
            }
        } while(flagDone && allRequests.size());
    }
    if(SpDebug::Controller.isEnable()){
        SpDebugPrint() << "[SpMpiBackgroundWorker] => worker stop";
    }
}
