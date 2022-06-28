#ifndef SPCONSUMERTHEAD_HPP
#define SPCONSUMERTHEAD_HPP

#include <thread>
#include <functional>
#include <future>
#include <queue>

class SpConsumerThread {
    static void consume(SpConsumerThread* data) {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(data->queueMutex);
                data->mutexCondition.wait(lock, [data] {
                    return !data->queueJobs.empty() || data->shouldTerminate;
                });
                if (data->shouldTerminate) {
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

public:
    SpConsumerThread()
        : thread(consume, this){

    }

    SpConsumerThread(const SpConsumerThread&) = delete;
    SpConsumerThread(SpConsumerThread&&) = delete;
    SpConsumerThread& operator=(const SpConsumerThread&) = delete;
    SpConsumerThread& operator=(SpConsumerThread&&) = delete;

    ~SpConsumerThread(){
        if(shouldTerminate == false){
            stop();
        }
    }

    void submitJob(const std::function<void()>& job) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueJobs.push(job);
        }
        mutexCondition.notify_one();
    }

    void submitJobAndWait(const std::function<void()>& job) {
        std::promise<int> synPromise;
        std::function<void()> superJob = [&job, &synPromise]() {
            job();
            synPromise.set_value(true);
        };
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueJobs.push(superJob);
        }
        mutexCondition.notify_one();
        synPromise.get_future().get();
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


#endif // SPCONSUMERTHEAD_HPP
