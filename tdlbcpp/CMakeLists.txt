cmake_minimum_required(VERSION 3.10)
project(tdlb LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

cmake_host_system_information(RESULT host_name QUERY HOSTNAME)

#Needed for Output and Checkpoint files
include (TestBigEndian)
TEST_BIG_ENDIAN(IS_BIG_ENDIAN)

find_package(BLAS REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_NO_SYSTEM_FROM_IMPORTED OFF)

set(HEADERS
	src/helper_cuda.h
	src/helper_string.h
	src/json.h
	src/cxxopts.hpp
	src/Header.h
	src/BaseParams.hpp
	src/GridParams.hpp
	src/FlowParams.hpp
	src/QVec.hpp
	src/ComputeUnit.h
	src/ComputeUnit.hpp
	src/CollisionEntropic.hpp
	src/CollisionEntropic.hpp
	src/StreamingNieve.hpp
	src/StreamingEsoteric.hpp
	src/Boundary.hpp
	src/Forcing.hpp
	src/Output.hpp
    tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCpp/GeomPolarLegacy.hpp
)

set(CPP
	src/main.cpp
)

option(WITH_CUDA "Build with CUDA support (default OFF)" OFF)

find_package(CUDA 10)

if (WITH_CUDA AND CUDA_VERSION)
    enable_language(CUDA)
    include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    set(GPU_CODE
        src/Collision.cu
        src/Streaming.cu
    )

    add_definitions(
        -D WITH_GPU=1
    )
else()
    add_definitions(
        -D WITH_GPU=0
    )
endif()

add_executable(${PROJECT_NAME} ${HEADERS} ${CPP} ${GPU_CODE})
source_group("GPU Code" FILES ${GPU_CODE})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE -DWITH_MPI=0 -DDEBUG)



#----------------------------------
# TESTS
#----------------------------------

#Needed to install googletest as part of project
#find_package(Git 1.6.5 REQUIRED)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
#add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
#                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
#                 EXCLUDE_FROM_ALL)



# Now simply link against gtest or gtest_main as needed. Eg
#add_executable(example example.cpp)
#target_link_libraries(example gtest_main)
#add_test(NAME example_test COMMAND example)




option(ENABLE_TESTS "Build and install tests (default ON)" OFF)

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory("${CMAKE_SOURCE_DIR}/tests")
endif()




