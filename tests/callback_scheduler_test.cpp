#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <eager_cancellation/callback_scheduler.hpp>
#include <lazy_canellation/callback_scheduler.hpp>

using namespace std::chrono_literals;

template <typename Scheduler, typename CallbackId>
void CancelAndExpect(Scheduler& scheduler, CallbackId id, bool expected) {
    using SchedulerT = std::decay_t<Scheduler>;
    if constexpr (std::is_same_v<SchedulerT, eager_cancellation::CallbackScheduler>) {
        EXPECT_EQ(expected, scheduler.Cancel(id));
    } else {
        scheduler.Cancel(id);
    }
}

using SchedulerTypes = ::testing::Types<
    lazy_cancellation::CallbackScheduler,
    eager_cancellation::CallbackScheduler
>;

template <typename Scheduler>
class CallbackSchedulerBasicTest : public ::testing::Test {};

TYPED_TEST_SUITE(CallbackSchedulerBasicTest, SchedulerTypes);

TYPED_TEST(CallbackSchedulerBasicTest, Simple)
{
    std::mutex m;
    std::condition_variable cv;
    bool done = false;

    auto callback = [&] {
        std::unique_lock lock(m);
        done = true;
        cv.notify_one();
    };

    TypeParam scheduler;
    auto when = std::chrono::system_clock::now() + 1s;
    scheduler.Schedule(std::move(callback), when);

    {
        std::unique_lock lock(m);
        cv.wait(lock, [&] { return done; });
    }
}

template <typename Scheduler>
class CallbackSchedulerFixture : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

public:
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

    Scheduler& getScheduler() {return scheduler_;}
private:
    std::mutex m_;
    std::condition_variable cv_;
    bool done_{false};

    Scheduler scheduler_;
};

TYPED_TEST_SUITE(CallbackSchedulerFixture, SchedulerTypes);

// ============================================================================
// SCHEDULING TESTS
// ============================================================================
TYPED_TEST(CallbackSchedulerFixture, SimpleTest)
{
    std::atomic_int val{0};
    auto inc = [&val] {++val;};
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    this->getScheduler().Schedule(inc,  now + 100ms);
    this->getScheduler().Schedule(inc,  now + 200ms);
    this->getScheduler().Schedule(sync, now + 300ms);
    this->getScheduler().Schedule(inc,  now + 400ms);
    this->WaitForDone();
    ASSERT_EQ(2, val) << "Last added inc shouldn't be executed at this moment";
    this->getScheduler().Schedule(sync, now + 500ms);
    this->WaitForDone();
    ASSERT_EQ(3, val) << "All incs should be executed at this moment";
}

TYPED_TEST(CallbackSchedulerFixture, MutualOrderTest) {
    std::atomic_int i{0};
    auto inc = [&i] () { ++i; };
    auto dec = [&i] () { --i; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    this->getScheduler().Schedule(inc, now + 100ms);
    this->getScheduler().Schedule(sync, now + 200ms);
    this->WaitForDone();
    EXPECT_EQ(1, i) << "The first scheduled callback increased val by one";
    this->getScheduler().Schedule(dec, now + 300ms);
    this->getScheduler().Schedule(sync, now + 400ms);
    this->WaitForDone();
    EXPECT_EQ(0, i) << "The second callback should decrease val by one";
}

TYPED_TEST(CallbackSchedulerFixture, SameTimeCallbacks) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto when = std::chrono::system_clock::now() + 100ms;
    this->getScheduler().Schedule(inc, when);
    this->getScheduler().Schedule(inc, when);
    this->getScheduler().Schedule(inc, when);
    this->getScheduler().Schedule(sync, when + 100ms);
    
    this->WaitForDone();
    EXPECT_EQ(3, counter) << "All three callbacks at same time should execute";
}

TYPED_TEST(CallbackSchedulerFixture, PastTimeCallback) {
    std::atomic_int val{0};
    auto inc = [&val] { ++val; };
    auto sync = [this] { this->SetDone(); };

    auto past = std::chrono::system_clock::now() - 1s;
    this->getScheduler().Schedule(inc, past);
    this->getScheduler().Schedule(sync, std::chrono::system_clock::now() + 100ms);
    
    this->WaitForDone();
    EXPECT_EQ(1, val) << "Past callback should execute immediately";
}

TYPED_TEST(CallbackSchedulerBasicTest, DestroyWithPendingCallbacks) {
    std::atomic_int executed{0};
    {
        TypeParam scheduler;
        auto far_future = std::chrono::system_clock::now() + 10s;
        scheduler.Schedule([&executed] { ++executed; }, far_future);
        scheduler.Schedule([&executed] { ++executed; }, far_future);
        scheduler.Schedule([&executed] { ++executed; }, far_future);
    }
    EXPECT_LE(executed, 3) << "Should not crash on destruction";
}

TYPED_TEST(CallbackSchedulerFixture, ManyCallbacks) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    for (int i = 0; i < 1'000; ++i) {
        this->getScheduler().Schedule(inc, now + std::chrono::milliseconds(10 + i));
    }
    this->getScheduler().Schedule(sync, now + 2s);
    
    this->WaitForDone();
    EXPECT_EQ(1'000, counter) << "All callbacks should execute";
}

TYPED_TEST(CallbackSchedulerFixture, ExecutionOrder) {
    std::vector<int> order;
    std::mutex order_mutex;
    
    auto record = [&](int n) {
        std::lock_guard lock(order_mutex);
        order.push_back(n);
    };
    
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    this->getScheduler().Schedule([&] { record(3); }, now + 300ms);
    this->getScheduler().Schedule([&] { record(1); }, now + 100ms);
    this->getScheduler().Schedule([&] { record(4); }, now + 400ms);
    this->getScheduler().Schedule([&] { record(2); }, now + 200ms);
    this->getScheduler().Schedule(sync, now + 500ms);
    
    this->WaitForDone();
    ASSERT_EQ(4, order.size());
    EXPECT_EQ(1, order[0]) << "First callback should be at 100ms";
    EXPECT_EQ(2, order[1]) << "Second callback should be at 200ms";
    EXPECT_EQ(3, order[2]) << "Third callback should be at 300ms";
    EXPECT_EQ(4, order[3]) << "Fourth callback should be at 400ms";
}

TYPED_TEST(CallbackSchedulerFixture, LongRunningCallback) {
    std::atomic_int counter{0};
    
    auto long_cb = [&] {
        std::this_thread::sleep_for(500ms);
    };

    auto now = std::chrono::system_clock::now();
    {
        TypeParam scheduler;
        scheduler.Schedule(long_cb, now);
    }
    SUCCEED() << "Long running callback shouldn't block scheduler";
}

TYPED_TEST(CallbackSchedulerBasicTest, EmptyScheduler) {
    {
        TypeParam scheduler;
    }
    SUCCEED() << "Empty scheduler should construct and destruct cleanly";
}

// ============================================================================
// CANCELLATION TESTS
// ============================================================================
TYPED_TEST(CallbackSchedulerFixture, CancelBeforeExecution) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto dec = [&counter] { --counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    this->getScheduler().Schedule(inc, now + 100ms);
    auto decId = this->getScheduler().Schedule(inc, now + 200ms);
    this->getScheduler().Schedule(inc, now + 300ms);
    
    CancelAndExpect(this->getScheduler(), decId, true);  // Cancel dec callback
    
    this->getScheduler().Schedule(sync, now + 400ms);
    this->WaitForDone();
    
    EXPECT_EQ(2, counter) << "Should execute 2 inc callbacks (dec was cancelled)";
}

TYPED_TEST(CallbackSchedulerFixture, CancelAll) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = this->getScheduler().Schedule(inc, now + 100ms);
    auto id2 = this->getScheduler().Schedule(inc, now + 200ms);
    auto id3 = this->getScheduler().Schedule(inc, now + 300ms);
    
    CancelAndExpect(this->getScheduler(), id1, true);
    CancelAndExpect(this->getScheduler(), id2, true);
    CancelAndExpect(this->getScheduler(), id3, true);
    
    this->getScheduler().Schedule(sync, now + 400ms);
    this->WaitForDone();
    
    EXPECT_EQ(0, counter) << "All callbacks were cancelled";
}

TYPED_TEST(CallbackSchedulerFixture, CancelEarliestCallback) {
    std::vector<int> order;
    std::mutex order_mutex;
    
    auto record = [&] (int n) {
        std::lock_guard lock(order_mutex);
        order.push_back(n);
    };
    
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = this->getScheduler().Schedule([&] { record(1); }, now + 100ms);
    this->getScheduler().Schedule([&] { record(2); }, now + 200ms);
    this->getScheduler().Schedule([&] { record(3); }, now + 300ms);
    
    CancelAndExpect(this->getScheduler(), id1, true);  // Cancel first callback
    
    this->getScheduler().Schedule(sync, now + 400ms);
    this->WaitForDone();
    
    ASSERT_EQ(2, order.size());
    EXPECT_EQ(2, order[0]) << "First executing callback should be #2";
    EXPECT_EQ(3, order[1]) << "Second executing callback should be #3";
}

TYPED_TEST(CallbackSchedulerFixture, CancelLatestCallback) {
    std::vector<int> order;
    std::mutex order_mutex;
    
    auto record = [&](int n) {
        std::lock_guard lock(order_mutex);
        order.push_back(n);
    };
    
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    this->getScheduler().Schedule([&] { record(1); }, now + 100ms);
    this->getScheduler().Schedule([&] { record(2); }, now + 200ms);
    auto id3 = this->getScheduler().Schedule([&] { record(3); }, now + 300ms);
    
    CancelAndExpect(this->getScheduler(), id3, true);  // Cancel last callback
    
    this->getScheduler().Schedule(sync, now + 400ms);
    this->WaitForDone();
    
    ASSERT_EQ(2, order.size());
    EXPECT_EQ(1, order[0]);
    EXPECT_EQ(2, order[1]);
}

TYPED_TEST(CallbackSchedulerFixture, CancelAfterExecution) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = this->getScheduler().Schedule(inc, now + 50ms);
    this->getScheduler().Schedule(sync, now + 100ms);
    
    this->WaitForDone();
    EXPECT_EQ(1, counter) << "Callback should have executed";
    
    // try to cancel after it executed
    CancelAndExpect(this->getScheduler(), id1, false);
    
    // no impact on callbacks added later on
    this->getScheduler().Schedule(inc, now + 200ms);
    this->getScheduler().Schedule(sync, now + 300ms);
    this->WaitForDone();
    
    EXPECT_EQ(2, counter) << "New callback should execute normally";
}

TYPED_TEST(CallbackSchedulerFixture, CancelNonExistentId) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    this->getScheduler().Schedule(inc, now + 100ms);
    
    CancelAndExpect(this->getScheduler(), 42, false);
    
    this->getScheduler().Schedule(sync, now + 200ms);
    this->WaitForDone();
    
    EXPECT_EQ(1, counter) << "Callback should still execute";
}

TYPED_TEST(CallbackSchedulerFixture, CancelMultipleTimes) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = this->getScheduler().Schedule(inc, now + 100ms);
    
    // cancel same id
    CancelAndExpect(this->getScheduler(), id1, true);
    CancelAndExpect(this->getScheduler(), id1, false);
    CancelAndExpect(this->getScheduler(), id1, false);
    
    this->getScheduler().Schedule(sync, now + 200ms);
    this->WaitForDone();
    
    EXPECT_EQ(0, counter) << "Callback should not execute (cancelled)";
}

TYPED_TEST(CallbackSchedulerFixture, CancelAndReschedule) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = this->getScheduler().Schedule(inc, now + 100ms);
    
    CancelAndExpect(this->getScheduler(), id1, true);
    
    // Schedule new callback (different ID)
    this->getScheduler().Schedule(inc, now + 150ms);
    this->getScheduler().Schedule(sync, now + 200ms);
    
    this->WaitForDone();
    EXPECT_EQ(1, counter) << "New callback should execute, cancelled one should not";
}

TYPED_TEST(CallbackSchedulerFixture, CancelAlternating) {
    std::vector<int> executed;
    std::mutex exec_mutex;
    
    auto record = [&](int n) {
        std::lock_guard lock(exec_mutex);
        executed.push_back(n);
    };
    
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = this->getScheduler().Schedule([&] { record(1); }, now + 100ms);
    auto id2 = this->getScheduler().Schedule([&] { record(2); }, now + 200ms);
    auto id3 = this->getScheduler().Schedule([&] { record(3); }, now + 300ms);
    auto id4 = this->getScheduler().Schedule([&] { record(4); }, now + 400ms);
    auto id5 = this->getScheduler().Schedule([&] { record(5); }, now + 500ms);
    
    // Cancel alternating callbacks
    CancelAndExpect(this->getScheduler(), id2, true);
    CancelAndExpect(this->getScheduler(), id4, true);
    
    this->getScheduler().Schedule(sync, now + 600ms);
    this->WaitForDone();
    
    ASSERT_EQ(3, executed.size());
    EXPECT_EQ(1, executed[0]);
    EXPECT_EQ(3, executed[1]);
    EXPECT_EQ(5, executed[2]);
}

TYPED_TEST(CallbackSchedulerFixture, CancelImmediateCallback) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    auto id1 = this->getScheduler().Schedule(inc, now + 50ms);
    
    CancelAndExpect(this->getScheduler(), id1, true);
    
    this->getScheduler().Schedule(sync, now + 150ms);
    this->WaitForDone();
    
    EXPECT_EQ(0, counter) << "Cancelled callback should not execute";
}

TYPED_TEST(CallbackSchedulerFixture, CancelManyCallbacks) {
    std::atomic_int counter{0};
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    std::vector<decltype(this->getScheduler().Schedule(inc, now))> ids;
    
    // schedule 100 callbacks
    for (int i = 0; i < 100; ++i) {
        ids.push_back(this->getScheduler().Schedule(inc, now + std::chrono::milliseconds(10 + i)));
    }
    
    // cancel every even
    for (size_t i = 0; i < ids.size(); i += 2) {
        CancelAndExpect(this->getScheduler(), ids[i], true);
    }
    
    this->getScheduler().Schedule(sync, now + 2s);
    this->WaitForDone();
    
    EXPECT_EQ(50, counter) << "50 callbacks should execute, 50 cancelled";
}

TYPED_TEST(CallbackSchedulerFixture, EarlierCallbackExecutesOnTime) {
    auto now = std::chrono::system_clock::now();
    auto later = now + 10s;
    auto earlier = now + 2s;

    this->getScheduler().Schedule([]{ /* far from now */ }, later);
    
    std::this_thread::sleep_for(1s);  // worker is waiting, 1 sec passed
    
    std::atomic_bool executed{false};
    this->getScheduler().Schedule([&executed] { executed = true; }, earlier);
    
    std::this_thread::sleep_for(2s); // +2 secs passed
    EXPECT_TRUE(executed) << "Early callback should have executed by now!";
}

TYPED_TEST(CallbackSchedulerFixture, MultipleCallbacksThrow) {
    std::atomic_int counter{0};
    auto do_throw = [] { throw std::runtime_error("Test exception"); };
    auto inc = [&counter] { ++counter; };
    auto sync = [this] { this->SetDone(); };

    auto now = std::chrono::system_clock::now();
    this->getScheduler().Schedule(do_throw, now + 100ms);
    this->getScheduler().Schedule(inc, now + 200ms);
    this->getScheduler().Schedule(do_throw, now + 300ms);
    this->getScheduler().Schedule(inc, now + 400ms);
    this->getScheduler().Schedule(do_throw, now + 500ms);
    this->getScheduler().Schedule(sync, now + 600ms);
    
    this->WaitForDone();
    
    EXPECT_EQ(2, counter) << "Non-throwing callbacks should execute despite exceptions";
}
