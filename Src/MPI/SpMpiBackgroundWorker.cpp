#include "SpMpiBackgroundWorker.hpp"
#include "Scheduler/SpTaskManager.hpp"


SpMpiBackgroundWorker SpMpiBackgroundWorker::MainWorker;


void SpMpiBackgroundWorker::Consume(SpMpiBackgroundWorker* data) {
    std::vector<MPI_Request> allRequests;
    std::vector<SpRequest> allRequestsTypes;

    std::unordered_map<int, SpMpiSendTransaction> sendTransactions;
    std::unordered_map<int, SpMpiRecvTransaction> recvTransactions;
    std::unordered_map<int, SpMpiBroadcastSendTransaction> broadcastSendTransactions;
    std::unordered_map<int, SpMpiBroadcastRecvTransaction> broadcastRecvTransactions;
    const int LimiteBroadcast = 1;

    int counterTransactions(0);

    while (true) {
        {
            std::unique_lock<std::mutex> lock(data->queueMutex);
            data->mutexCondition.wait(lock, [data, &allRequests] {
                return !data->newSends.empty()
                        || !data->newRecvs.empty()
                        || !allRequests.empty()
                        || !data->newBroadcastSends.empty()
                        || !data->newBroadcastSends.empty()
                        || data->shouldTerminate;
            });
            if (data->shouldTerminate) {
                assert(data->newSends.empty()
                       && data->newRecvs.empty()
                       && sendTransactions.empty()
                       && recvTransactions.empty()
                       && allRequests.empty()
                       && broadcastSendTransactions.empty()
                       && broadcastRecvTransactions.empty());
                break;
            }
            while(!data->newSends.empty()){
                auto func = std::move(data->newSends.back());
                data->newSends.pop_back();
                sendTransactions[counterTransactions] = func();

                SpMpiSendTransaction& tr = sendTransactions[counterTransactions];
                allRequestsTypes.emplace_back(SpRequest{TYPE_SEND, STATE_FIRST, counterTransactions});
                allRequests.emplace_back(tr.requestBufferSize);
                allRequestsTypes.emplace_back(SpRequest{TYPE_SEND, STATE_SECOND, counterTransactions});
                allRequests.emplace_back(tr.request);
                counterTransactions += 1;
            }
            while(!data->newRecvs.empty()){
                auto func = data->newRecvs.back();
                data->newRecvs.pop_back();
                recvTransactions[counterTransactions] = func();

                SpMpiRecvTransaction& tr = recvTransactions[counterTransactions];
                allRequestsTypes.emplace_back(SpRequest{TYPE_RECV, STATE_FIRST, counterTransactions});
                allRequests.emplace_back(tr.requestBufferSize);
                counterTransactions += 1;
            }
            while(!data->newBroadcastRecvs.empty() && broadcastRecvTransactions.size() < LimiteBroadcast){
                auto func = std::move(data->newBroadcastRecvs.front());
                data->newBroadcastRecvs.pop();
                broadcastRecvTransactions[counterTransactions] = func();

                SpMpiBroadcastRecvTransaction& tr = broadcastRecvTransactions[counterTransactions];
                allRequestsTypes.emplace_back(SpRequest{TYPE_BROADCASTRECV, STATE_FIRST, counterTransactions});
                allRequests.emplace_back(tr.requestBufferSize);
                counterTransactions += 1;
            }
            while(!data->newBroadcastSends.empty() && broadcastSendTransactions.size() < LimiteBroadcast){
                auto func = std::move(data->newBroadcastSends.front());
                data->newBroadcastSends.pop();
                broadcastSendTransactions[counterTransactions] = func();

                SpMpiBroadcastSendTransaction& tr = broadcastSendTransactions[counterTransactions];
                allRequestsTypes.emplace_back(SpRequest{TYPE_BROADCASTSEND, STATE_FIRST, counterTransactions});
                allRequests.emplace_back(tr.requestBufferSize);
                allRequestsTypes.emplace_back(SpRequest{TYPE_BROADCASTSEND, STATE_SECOND, counterTransactions});
                allRequests.emplace_back(tr.request);
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
                SpRequest rt = allRequestsTypes[idxDone];
                std::swap(allRequestsTypes[idxDone], allRequestsTypes.back());
                allRequestsTypes.pop_back();
                std::swap(allRequests[idxDone], allRequests.back());
                allRequests.pop_back();

                if(rt.type == TYPE_SEND){
                    assert(sendTransactions.find(rt.idxTransaction) != sendTransactions.end());
                    if(SpDebug::Controller.isEnable()){
                        SpDebugPrint() << "[SpMpiBackgroundWorker] => send done " << rt.idxTransaction;
                    }
                    if(rt.state == STATE_SECOND){
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
                else if(rt.type == TYPE_RECV){
                    assert(recvTransactions.find(rt.idxTransaction) != recvTransactions.end());
                    if(rt.state == STATE_FIRST){
                        if(SpDebug::Controller.isEnable()){
                            SpDebugPrint() << "[SpMpiBackgroundWorker] => recv state 0 " << rt.idxTransaction;
                        }
                        // Size recv
                        SpMpiRecvTransaction& transaction = recvTransactions[rt.idxTransaction];
                        transaction.buffer.resize(*transaction.bufferSize);
                        transaction.request = Irecv(transaction.buffer.data(),
                                                      int(transaction.buffer.size()),
                                                      transaction.srcProc, transaction.tag, data->mpiCom);
                        allRequestsTypes.emplace_back(SpRequest{TYPE_RECV, STATE_SECOND, rt.idxTransaction});
                        allRequests.emplace_back(transaction.request);
                    }
                    else if(rt.state == STATE_SECOND){
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
                else if(rt.type == TYPE_BROADCASTSEND){
                    {// TODO
                        std::cout << SpMpiUtils::GetMpiRank() << " SEND "  << " rt.idxTransaction = " << rt.idxTransaction << " rt.state " << rt.state << std::endl;
                        for(auto& va: broadcastSendTransactions){
                            std::cout << SpMpiUtils::GetMpiRank() << " " << va.first << std::endl;
                        }
                        std::cout << SpMpiUtils::GetMpiRank() << " rt.idxTransaction = " << rt.idxTransaction << std::endl;
                    }
                    assert(broadcastSendTransactions.find(rt.idxTransaction) != broadcastSendTransactions.end());
                    if(SpDebug::Controller.isEnable()){
                        SpDebugPrint() << "[SpMpiBackgroundWorker] => broadcast send done " << rt.idxTransaction;
                    }
                    if(rt.state == STATE_SECOND){
                        if(SpDebug::Controller.isEnable()){
                            SpDebugPrint() << "[SpMpiBackgroundWorker] => broadcast send complete " << rt.idxTransaction;
                        }
                        // Send done
                        SpMpiBroadcastSendTransaction transaction = std::move(broadcastSendTransactions[rt.idxTransaction]);
                        broadcastSendTransactions.erase(rt.idxTransaction);
                        // Post back task
                        transaction.tm->postMPITaskExecution(*transaction.atg, transaction.task);
                        transaction.releaseRequest();
                    }
                }
                else if(rt.type == TYPE_BROADCASTRECV){                    
                    {// TODO
                        std::cout << SpMpiUtils::GetMpiRank() << " RECV "  << " rt.idxTransaction = " << rt.idxTransaction << " rt.state " << rt.state << std::endl;
                        for(auto& va: broadcastRecvTransactions){
                            std::cout << SpMpiUtils::GetMpiRank() << " " << va.first << std::endl;
                        }
                        std::cout << SpMpiUtils::GetMpiRank() << " rt.idxTransaction = " << rt.idxTransaction << std::endl;
                    }
                    assert(broadcastRecvTransactions.find(rt.idxTransaction) != broadcastRecvTransactions.end());
                    if(rt.state == STATE_FIRST){
                        if(SpDebug::Controller.isEnable()){
                            SpDebugPrint() << "[SpMpiBackgroundWorker] => broadcast recv state 0 " << rt.idxTransaction;
                        }
                        // Size recv
                        SpMpiBroadcastRecvTransaction& transaction = broadcastRecvTransactions[rt.idxTransaction];
                        transaction.buffer.resize(*transaction.bufferSize);
                        transaction.request = IBroadcastrecv(transaction.buffer.data(),
                                                      int(transaction.buffer.size()),
                                                      transaction.root, data->mpiCom);
                        allRequestsTypes.emplace_back(SpRequest{TYPE_BROADCASTRECV, STATE_SECOND, rt.idxTransaction});
                        allRequests.emplace_back(transaction.request);
                    }
                    else if(rt.state == STATE_SECOND){
                        if(SpDebug::Controller.isEnable()){
                            SpDebugPrint() << "[SpMpiBackgroundWorker] => broadcast recv state 1 " << rt.idxTransaction;
                        }
                        // Recv done
                        SpMpiBroadcastRecvTransaction transaction = std::move(broadcastRecvTransactions[rt.idxTransaction]);
                        broadcastRecvTransactions.erase(rt.idxTransaction);
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
