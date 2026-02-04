include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
)

FetchContent_MakeAvailable(googletest)

add_executable(callback_scheduler_test
    callback_scheduler_test.cpp
)

target_link_libraries(callback_scheduler_test
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(callback_scheduler_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

include(GoogleTest)
gtest_discover_tests(callback_scheduler_test)
