#
# @file    CMakeLists.txt
# @brief   Configuration for the project
# @author  Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date    2019/07/09
#

# ----------------- PRELIMINARIES ---------------------------------------------

# jmbuena: Set this variable to the dir containing include/gtest/getst.h file.
set(GTEST_ROOT_DIR "/usr/include/gtest")

# Build with tests
option(WITH_TESTS "Build tests enabled" ON)

# ----------------- BUILD A LIBRARY -------------------------------------------

# Create the object detection library
add_library(${PROJECT_NAME} STATIC)

# Library header files
target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
#message(STATUS  "PROJECT_NAME=${PROJECT_NAME}")
#message(STATUS  "INCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/include")
##message(STATUS  "INCLUDE_DIRECTORIES = $<TARGET_PROPERTY:${PROJECT_NAME},INCLUDE_DIRECTORIES>")

# ----------------- OTHER LIBRARIES -------------------------------------------

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_VERSION=${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
message(STATUS "OpenCV_DIR=${OpenCV_DIR}")

# OpenCV headers
target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)

# Link OpenCV libraries
target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
)

# Pthreads
find_package(Threads)
target_link_libraries(
        ${PROJECT_NAME}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
)

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)


### ----------------- CPP CONFIGURATION -----------------------------------------

# Allows generated files in the current directed to be found by the compiler
#SET(CMAKE_INCLUDE_CURRENT_DIR ON)
# Remove deprecation warnings
set(CMAKE_WARN_DEPRECATED FALSE)
# Full compiler output
set(CMAKE_VERBOSE_MAKEFILE ON)
# C++11 and beyond
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------- LIBRARY COMPILATION CONFIGURATION -------------------------

# Library source files
add_subdirectory(src)

# Some binaries to build
add_subdirectory(bin)

## Build the library with tests.
if( WITH_TESTS )
  # Enable test-dependent macros
  add_definitions(-DWITH_TESTS)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_TESTS=1)

  # Find Google tests
  find_package(GTest REQUIRED)
  include_directories(${GTEST_BOTH_LIBRARIES})

  set(GTEST_ROOT GTEST_ROOT_DIR ${GTEST_ROOT_DIR})
  find_package(GTest)
  if(NOT GTEST_LIBRARY)
    message(WARNING "gtest not found, download and compile it!")
    return()
  endif()

  # parse tests directory
  enable_testing() # Enable testing using 'make test'
  add_subdirectory(tests)
endif( WITH_TESTS )



