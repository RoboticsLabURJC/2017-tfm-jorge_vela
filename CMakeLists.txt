
project(objdetect)

cmake_minimum_required(VERSION 3.14)

add_subdirectory(obj_detect)

# ----------------- OTHER LIBRARIES -------------------------------------------

## OpenCV
find_package(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV_VERSION=${OpenCV_VERSION}")
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
MESSAGE(STATUS "OpenCV_DIR=${OpenCV_DIR}")

## OpenCV headers
#TARGET_INCLUDE_DIRECTORIES(
#        ${PROJECT_NAME}
#        PUBLIC
#        ${OpenCV_INCLUDE_DIRS}
#)

## Link OpenCV libraries
#target_link_libraries(
#        ${PROJECT_NAME}
#        ${OpenCV_LIBS}
#)

# Pthreads
#find_package(Threads)
#target_link_libraries(
#        ${PROJECT_NAME}
#        ${CMAKE_THREAD_LIBS_INIT}
#        ${CMAKE_DL_LIBS}
#)


#find_package(OpenMP)
#if(OPENMP_FOUND)
#  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif(OPENMP_FOUND)


# ----------------- CPP CONFIGURATION -----------------------------------------

# Allows generated files in the current directed to be found by the compiler
# SET(CMAKE_INCLUDE_CURRENT_DIR ON)
# Remove deprecation warnings
set(CMAKE_WARN_DEPRECATED FALSE)
# Full compiler output
set(CMAKE_VERBOSE_MAKEFILE ON)
# C++11 and beyond
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------- testProgram -----------------------------------------
add_executable(app testProgram.cpp)
target_include_directories(
        app
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/obj_detect/include"
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
       app
       ${PROJECT_NAME}
       ${OpenCV_LIBS}
)

# ----------------- test_mat -----------------------------------------
add_executable(test_mat test_mat.cpp)
TARGET_INCLUDE_DIRECTORIES(
        test_mat
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
       test_mat
       ${OpenCV_LIBS}
)

# ----------------- test_umat -----------------------------------------
add_executable(test_umat test_umat.cpp)
target_include_directories(
        test_umat
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
       test_umat
       ${OpenCV_LIBS}
)


# ------------------ computeDetectionFaces ----------------------------
add_executable(computeDetFaces computeDetectionFaces.cpp)
target_include_directories(computeDetFaces
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/obj_detect/include"
)
target_link_libraries(computeDetFaces ${PROJECT_NAME})

# ------------------ computeDetectionCars -----------------------------
add_executable(computeDetCars computeDetectionCars.cpp)
target_include_directories(computeDetCars
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/obj_detect/include"
)
target_link_libraries(computeDetCars ${PROJECT_NAME})
