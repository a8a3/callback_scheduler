#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_set>

namespace lazy_cancellation {

using TimePoint = std::chrono::time_point<std::chrono::system_clock>;
using Callback = std::function<void()>;
using CallbackId = uint64_t;

struct ScheduledCallback{
    CallbackId id;
    Callback cb;
    TimePoint when;

    friend bool operator<(const ScheduledCallback& a, const ScheduledCallback& b) {
        return a.when > b.when;
    }
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

                auto wakeTime = scheduledCallbacks_.top().when;
                cv_.wait_until(lock, wakeTime);
                if (!isRunning_) return;
                if (scheduledCallbacks_.empty()) continue;

                const auto& next = scheduledCallbacks_.top();
                if (next.when <= std::chrono::system_clock::now()) {
                    auto cb = next.cb;
                    auto id = next.id;
                    scheduledCallbacks_.pop();

                    if (cancelledCallbacks_.count(id)) {
                        cancelledCallbacks_.erase(id);
                        continue;
                    }
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
            needToNotify = scheduledCallbacks_.empty() || scheduledCallbacks_.top().when > when;
            scheduledCallbacks_.emplace(id, std::move(callback), when);
        }
        if (needToNotify) cv_.notify_one();
        return id;
    }

    void Cancel(CallbackId id) {
        std::lock_guard lock{mutex_};
        cancelledCallbacks_.insert(id);
    }

private:
    std::priority_queue<ScheduledCallback> scheduledCallbacks_;
    std::thread worker_;  // thread in context of which callbacks are executed
    std::atomic_bool isRunning_{true};
    std::mutex mutex_;
    std::condition_variable cv_;

    // NB! those two always used with mutex held
    CallbackId nextId_{0};
    std::unordered_set<CallbackId> cancelledCallbacks_;
};

} // namespace lazy_cancellation
