cmake_minimum_required(VERSION 3.14.0)
project(ai VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(include)
# include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_executable(ai src/main.cpp)

target_link_libraries(ai PUBLIC ai_lib)

# test ---------------------------------------------------------------------------------

add_definitions(-DUNIT_TEST) # enable public interfaces to test

# add gtest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
    
enable_testing()

add_executable(
  tests
  test/test.cpp
)
target_link_libraries(
  tests
  ai_lib
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)

# --------------------------------------------------------------------------------------

# include(CTest)
# enable_testing()


# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
