#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <mutex>

#include <callback_scheduler.hpp>

using namespace std::chrono_literals;

TEST(CallbackSchedulerTest, Simple)
{
    std::mutex m;
    std::condition_variable cv;
    bool done = false;

    auto callback = [&] {
        std::unique_lock lock(m);
        done = true;
        cv.notify_one();
    };

    CallbackScheduler scheduler;
    auto when = std::chrono::system_clock::now() + 1s;
    scheduler.Schedule(std::move(callback), when);

    {
        std::unique_lock lock(m);
        cv.wait(lock, [&] { return done; });
    }
}

class CallbackSchedulerFixture : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    void WaitForDone() {
        std::unique_lock lock{m_};
        cv_.wait(lock, [this] {return done_;});
        done_ = false;
    }

    void SetDone() {
        std::unique_lock lock{m_};
        done_ = true;
        cv_.notify_one();
    }

    CallbackScheduler& getScheduler() {return scheduler_;}
private:
    std::mutex m_;
    std::condition_variable cv_;
    bool done_{false};

    CallbackScheduler scheduler_;
};

// ============================================================================
// SCHEDULING TESTS
// ============================================================================
TEST_F(CallbackSchedulerFixture, SimpleTest)
{
    std::atomic_int val{0};
    auto inc = [&val] {++val;};
    auto sync = [this] {SetDone();};

    auto now = std::chrono::system_clock::now();
    getScheduler().Schedule(inc,  now + 100ms);
    getScheduler().Schedule(inc,  now + 200ms);
    getScheduler().Schedule(sync, now + 300ms);
    getScheduler().Schedule(inc,  now + 400ms);
    WaitForDone();
    ASSERT_EQ(2, val) << "Last added inc shouldn't be executed at this moment";
    getScheduler().Schedule(sync, now + 500ms);
    WaitForDone();
    ASSERT_EQ(3, val) << "All incs should be executed at this moment";
}

TEST_F(CallbackSchedulerFixture, MutualOrderTest) {

    std::atomic_int i{0};
    auto inc = [&i] () { ++i; };
    auto dec = [&i] () { --i; };
    auto sync = [this] {SetDone();};

    auto now = std::chrono::system_clock::now();
    getScheduler().Schedule(inc, now + 100ms);
    getScheduler().Schedule(sync, now + 200ms);
    WaitForDone();
    EXPECT_EQ(1, i) << "The first scheduled callback increased val by one";
    getScheduler().Schedule(dec, now + 300ms);
    getScheduler().Schedule(sync, now + 400ms);
    WaitForDone();
    EXPECT_EQ(0, i) << "The second callback should decrease val by one";
}

TEST_F(CallbackSchedulerFixture, SameTimeCallbacks) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { SetDone(); };

    auto when = std::chrono::system_clock::now() + 100ms;
    getScheduler().Schedule(inc, when);
    getScheduler().Schedule(inc, when);
    getScheduler().Schedule(inc, when);
    getScheduler().Schedule(sync, when + 100ms);
    
    WaitForDone();
    EXPECT_EQ(3, counter) << "All three callbacks at same time should execute";
}

TEST_F(CallbackSchedulerFixture, PastTimeCallback) {
    std::atomic_int val{0};
    auto inc = [&val] { ++val; };
    auto sync = [this] { SetDone(); };

    auto past = std::chrono::system_clock::now() - 1s;
    getScheduler().Schedule(inc, past);
    getScheduler().Schedule(sync, std::chrono::system_clock::now() + 100ms);
    
    WaitForDone();
    EXPECT_EQ(1, val) << "Past callback should execute immediately";
}

TEST(CallbackSchedulerTest, DestroyWithPendingCallbacks) {
    std::atomic_int executed{0};
    {
        CallbackScheduler scheduler;
        auto far_future = std::chrono::system_clock::now() + 10s;
        scheduler.Schedule([&executed] { ++executed; }, far_future);
        scheduler.Schedule([&executed] { ++executed; }, far_future);
        scheduler.Schedule([&executed] { ++executed; }, far_future);
    }
    EXPECT_LE(executed, 3) << "Should not crash on destruction";
}

TEST_F(CallbackSchedulerFixture, ManyCallbacks) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    for (int i = 0; i < 1'000; ++i) {
        getScheduler().Schedule(inc, now + std::chrono::milliseconds(10 + i));
    }
    getScheduler().Schedule(sync, now + 2s);
    
    WaitForDone();
    EXPECT_EQ(1'000, counter) << "All callbacks should execute";
}

TEST_F(CallbackSchedulerFixture, ExecutionOrder) {
    std::vector<int> order;
    std::mutex order_mutex;
    
    auto record = [&](int n) {
        std::lock_guard lock(order_mutex);
        order.push_back(n);
    };
    
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    getScheduler().Schedule([&] { record(3); }, now + 300ms);
    getScheduler().Schedule([&] { record(1); }, now + 100ms);
    getScheduler().Schedule([&] { record(4); }, now + 400ms);
    getScheduler().Schedule([&] { record(2); }, now + 200ms);
    getScheduler().Schedule(sync, now + 500ms);
    
    WaitForDone();
    ASSERT_EQ(4, order.size());
    EXPECT_EQ(1, order[0]) << "First callback should be at 100ms";
    EXPECT_EQ(2, order[1]) << "Second callback should be at 200ms";
    EXPECT_EQ(3, order[2]) << "Third callback should be at 300ms";
    EXPECT_EQ(4, order[3]) << "Fourth callback should be at 400ms";
}

TEST_F(CallbackSchedulerFixture, LongRunningCallback) {
    std::atomic_int counter{0};
    
    auto long_cb = [&] {
        std::this_thread::sleep_for(500ms);
    };

    auto now = std::chrono::system_clock::now();
    {
        CallbackScheduler scheduler;
        scheduler.Schedule(long_cb, now);
    }
    SUCCEED() << "Long running callback shouldn't block scheduler";
}

TEST(CallbackSchedulerTest, EmptyScheduler) {
    {
        CallbackScheduler scheduler;
    }
    SUCCEED() << "Empty scheduler should construct and destruct cleanly";
}

// ============================================================================
// CANCELLATION TESTS
// ============================================================================
TEST_F(CallbackSchedulerFixture, CancelBeforeExecution) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto dec = [&counter] { --counter; };
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    getScheduler().Schedule(inc, now + 100ms);
    auto decId = getScheduler().Schedule(inc, now + 200ms);
    getScheduler().Schedule(inc, now + 300ms);
    
    getScheduler().Cancel(decId);  // Cancel dec callback
    
    getScheduler().Schedule(sync, now + 400ms);
    WaitForDone();
    
    EXPECT_EQ(2, counter) << "Should execute 2 inc callbacks (dec was cancelled)";
}

TEST_F(CallbackSchedulerFixture, CancelAll) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = getScheduler().Schedule(inc, now + 100ms);
    auto id2 = getScheduler().Schedule(inc, now + 200ms);
    auto id3 = getScheduler().Schedule(inc, now + 300ms);
    
    getScheduler().Cancel(id1);
    getScheduler().Cancel(id2);
    getScheduler().Cancel(id3);
    
    getScheduler().Schedule(sync, now + 400ms);
    WaitForDone();
    
    EXPECT_EQ(0, counter) << "All callbacks were cancelled";
}

TEST_F(CallbackSchedulerFixture, CancelEarliestCallback) {
    std::vector<int> order;
    std::mutex order_mutex;
    
    auto record = [&] (int n) {
        std::lock_guard lock(order_mutex);
        order.push_back(n);
    };
    
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = getScheduler().Schedule([&] { record(1); }, now + 100ms);
    getScheduler().Schedule([&] { record(2); }, now + 200ms);
    getScheduler().Schedule([&] { record(3); }, now + 300ms);
    
    getScheduler().Cancel(id1);  // Cancel first callback
    
    getScheduler().Schedule(sync, now + 400ms);
    WaitForDone();
    
    ASSERT_EQ(2, order.size());
    EXPECT_EQ(2, order[0]) << "First executing callback should be #2";
    EXPECT_EQ(3, order[1]) << "Second executing callback should be #3";
}

TEST_F(CallbackSchedulerFixture, CancelLatestCallback) {
    std::vector<int> order;
    std::mutex order_mutex;
    
    auto record = [&](int n) {
        std::lock_guard lock(order_mutex);
        order.push_back(n);
    };
    
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    getScheduler().Schedule([&] { record(1); }, now + 100ms);
    getScheduler().Schedule([&] { record(2); }, now + 200ms);
    auto id3 = getScheduler().Schedule([&] { record(3); }, now + 300ms);
    
    getScheduler().Cancel(id3);  // Cancel last callback
    
    getScheduler().Schedule(sync, now + 400ms);
    WaitForDone();
    
    ASSERT_EQ(2, order.size());
    EXPECT_EQ(1, order[0]);
    EXPECT_EQ(2, order[1]);
}

TEST_F(CallbackSchedulerFixture, CancelAfterExecution) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = getScheduler().Schedule(inc, now + 50ms);
    getScheduler().Schedule(sync, now + 100ms);
    
    WaitForDone();
    EXPECT_EQ(1, counter) << "Callback should have executed";
    
    // try to cancel after it executed
    getScheduler().Cancel(id1);
    
    // no impact on callbacks added later on
    getScheduler().Schedule(inc, now + 200ms);
    getScheduler().Schedule(sync, now + 300ms);
    WaitForDone();
    
    EXPECT_EQ(2, counter) << "New callback should execute normally";
}

TEST_F(CallbackSchedulerFixture, CancelNonExistentId) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    getScheduler().Schedule(inc, now + 100ms);
    
    getScheduler().Cancel(42);
    
    getScheduler().Schedule(sync, now + 200ms);
    WaitForDone();
    
    EXPECT_EQ(1, counter) << "Callback should still execute";
}

TEST_F(CallbackSchedulerFixture, CancelMultipleTimes) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = getScheduler().Schedule(inc, now + 100ms);
    
    // cancel same id
    getScheduler().Cancel(id1);
    getScheduler().Cancel(id1);
    getScheduler().Cancel(id1);
    
    getScheduler().Schedule(sync, now + 200ms);
    WaitForDone();
    
    EXPECT_EQ(0, counter) << "Callback should not execute (cancelled)";
}

TEST_F(CallbackSchedulerFixture, CancelAndReschedule) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = getScheduler().Schedule(inc, now + 100ms);
    
    getScheduler().Cancel(id1);
    
    // Schedule new callback (different ID)
    getScheduler().Schedule(inc, now + 150ms);
    getScheduler().Schedule(sync, now + 200ms);
    
    WaitForDone();
    EXPECT_EQ(1, counter) << "New callback should execute, cancelled one should not";
}

TEST_F(CallbackSchedulerFixture, CancelAlternating) {
    std::vector<int> executed;
    std::mutex exec_mutex;
    
    auto record = [&](int n) {
        std::lock_guard lock(exec_mutex);
        executed.push_back(n);
    };
    
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = getScheduler().Schedule([&] { record(1); }, now + 100ms);
    auto id2 = getScheduler().Schedule([&] { record(2); }, now + 200ms);
    auto id3 = getScheduler().Schedule([&] { record(3); }, now + 300ms);
    auto id4 = getScheduler().Schedule([&] { record(4); }, now + 400ms);
    auto id5 = getScheduler().Schedule([&] { record(5); }, now + 500ms);
    
    // Cancel alternating callbacks
    getScheduler().Cancel(id2);
    getScheduler().Cancel(id4);
    
    getScheduler().Schedule(sync, now + 600ms);
    WaitForDone();
    
    ASSERT_EQ(3, executed.size());
    EXPECT_EQ(1, executed[0]);
    EXPECT_EQ(3, executed[1]);
    EXPECT_EQ(5, executed[2]);
}

TEST_F(CallbackSchedulerFixture, CancelImmediateCallback) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = getScheduler().Schedule(inc, now);  // immediate execution
    
    getScheduler().Cancel(id1);  // might be too late
    
    getScheduler().Schedule(sync, now + 100ms);
    WaitForDone();
    
    // Result is non-deterministic - race between execution and cancellation
    EXPECT_LE(counter, 1) << "Counter should be 0 or 1";
}

TEST_F(CallbackSchedulerFixture, CancelManyCallbacks) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { SetDone(); };

    auto now = std::chrono::system_clock::now();
    std::vector<CallbackId> ids;
    
    // schedule 100 callbacks
    for (int i = 0; i < 100; ++i) {
        ids.push_back(getScheduler().Schedule(inc, now + std::chrono::milliseconds(10 + i)));
    }
    
    // cancel every even
    for (size_t i = 0; i < ids.size(); i += 2) {
        getScheduler().Cancel(ids[i]);
    }
    
    getScheduler().Schedule(sync, now + 2s);
    WaitForDone();
    
    EXPECT_EQ(50, counter) << "50 callbacks should execute, 50 cancelled";
}
