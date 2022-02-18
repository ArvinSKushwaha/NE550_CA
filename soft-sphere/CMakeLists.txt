project(SoftSphere)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_executable(${PROJECT_NAME} main.cpp)
