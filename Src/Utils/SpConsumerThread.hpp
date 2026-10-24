#ifndef SPCONSUMERTHEAD_HPP
#define SPCONSUMERTHEAD_HPP

#include <thread>
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <memory>

class SpConsumerThread {
    static void consume(SpConsumerThread* data) {
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
                job = std::move(data->queueJobs.front());
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
    const std::thread::id consumerThreadId;

public:
    SpConsumerThread()
        : thread(consume, this), consumerThreadId(thread.get_id()){

    }

    SpConsumerThread(const SpConsumerThread&) = delete;
    SpConsumerThread(SpConsumerThread&&) = delete;
    SpConsumerThread& operator=(const SpConsumerThread&) = delete;
    SpConsumerThread& operator=(SpConsumerThread&&) = delete;

    ~SpConsumerThread(){
        stop();
    }

    void submitJob(const std::function<void()>& job) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if(shouldTerminate) {
                throw std::runtime_error("Cannot submit a job to a stopped SpConsumerThread");
            }
            queueJobs.push(job);
        }
        mutexCondition.notify_one();
    }

    void submitJobAndWait(const std::function<void()>& job) {
        // A synchronous submission from this consumer must run inline to
        // avoid waiting on the thread that is currently executing the caller.
        if(std::this_thread::get_id() == consumerThreadId) {
            job();
            return;
        }
        auto task = std::make_shared<std::packaged_task<void()>>(job);
        auto result = task->get_future();
        submitJob([task]() { (*task)(); });
        result.get();
    }

    void stop() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            shouldTerminate = true;
        }
        mutexCondition.notify_all();
        // stop() calls must be serialized by the caller. A consumer job may
        // request shutdown; an external stop/destructor performs the join.
        if(std::this_thread::get_id() != consumerThreadId && thread.joinable()) {
            thread.join();
        }
    }
};


#endif // SPCONSUMERTHEAD_HPP
