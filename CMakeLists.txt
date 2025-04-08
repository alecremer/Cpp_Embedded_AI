cmake_minimum_required(VERSION 3.10.0)
project(ai VERSION 0.1.0 LANGUAGES C CXX)


add_subdirectory(include)
# include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_executable(ai src/main.cpp)

target_link_libraries(ai PUBLIC ai_lib)

include(CTest)
enable_testing()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
