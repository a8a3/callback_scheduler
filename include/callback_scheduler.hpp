#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

using TimePoint = std::chrono::time_point<std::chrono::system_clock>;
using Callback = std::function<void()>;
using CallbackId = uint64_t;

struct ScheduledCallback{
    CallbackId id;
    Callback cb;
};

class CallbackScheduler {
public:
    CallbackScheduler() {
        worker_ = std::thread{[this] () {
            while (isRunning_) {
                std::unique_lock lock{mutex_};
                if (scheduledCallbacks_.empty()) {
                    cv_.wait(lock, [this] {
                        return !scheduledCallbacks_.empty() || !isRunning_;
                    });
                    if (!isRunning_) return;
                    continue;
                }

                const auto wakeTime = scheduledCallbacks_.begin()->first;
                cv_.wait_until(lock, wakeTime);
                if (!isRunning_) return;
                if (scheduledCallbacks_.empty()) continue;

                const auto next = scheduledCallbacks_.begin();
                if (next->first <= std::chrono::system_clock::now()) {
                    auto cb = next->second.cb;
                    auto id = next->second.id;
                    scheduledCallbacks_.erase(next);
                    idMap_.erase(id);

                    lock.unlock();

                    try {
                        cb();
                    } catch (...) {
                        // do something
                    }
                }
            }
        }};
    }

    ~CallbackScheduler() {
        {
            std::lock_guard lock{mutex_};
            isRunning_ = false;
        }
        cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    CallbackId Schedule(Callback callback, TimePoint when) {
        CallbackId id{0};
        bool needToNotify{false};
        {
            std::lock_guard lock{mutex_};
            id = nextId_++;
            needToNotify = scheduledCallbacks_.empty() || scheduledCallbacks_.begin()->first > when;
            auto cbIter = scheduledCallbacks_.emplace(std::make_pair(when, ScheduledCallback{id, std::move(callback)}));
            idMap_.emplace(std::make_pair(id, cbIter));
        }
        if (needToNotify) cv_.notify_one();
        return id;
    }

    bool Cancel(CallbackId id) {
        std::lock_guard lock{mutex_};
        auto cbIter = idMap_.find(id);

        if (std::end(idMap_) == cbIter) return false;
        scheduledCallbacks_.erase(cbIter->second);
        idMap_.erase(cbIter);
        return true;
    }

private:
    std::multimap<TimePoint, ScheduledCallback> scheduledCallbacks_;
    std::unordered_map<CallbackId, decltype(scheduledCallbacks_)::iterator> idMap_;

    std::thread worker_;  // thread in context of which callbacks are executed
    std::atomic_bool isRunning_{true};
    std::mutex mutex_;
    std::condition_variable cv_;

    CallbackId nextId_{0};
};
