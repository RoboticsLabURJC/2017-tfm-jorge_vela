
cmake_minimum_required(VERSION 3.2)

project(objdetect)
add_subdirectory(obj_detect)

## OpenCV
find_package(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV_VERSION=${OpenCV_VERSION}")
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
MESSAGE(STATUS "OpenCV_DIR=${OpenCV_DIR}")

ADD_EXECUTABLE(app testProgram.cpp)
TARGET_INCLUDE_DIRECTORIES(
        app
        PUBLIC
        obj_detect/include
        ${OpenCV_INCLUDE_DIRS}
)
TARGET_LINK_LIBRARIES(
       app
       ${PROJECT_NAME}
       ${OpenCV_LIBS}
)


ADD_EXECUTABLE(test_mat test_mat.cpp)
TARGET_INCLUDE_DIRECTORIES(
        test_mat
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)
TARGET_LINK_LIBRARIES(
       test_mat
       ${OpenCV_LIBS}
)

ADD_EXECUTABLE(test_umat test_umat.cpp)
TARGET_INCLUDE_DIRECTORIES(
        test_umat
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)
TARGET_LINK_LIBRARIES(
       test_umat
       ${OpenCV_LIBS}
)


add_executable(computeDetFaces computeDetectionFaces.cpp)

target_include_directories(computeDetFaces PUBLIC
  obj_detect/include 
)

add_executable(computeDetCars computeDetectionCars.cpp)

target_include_directories(computeDetCars PUBLIC
  obj_detect/include 
)

target_link_libraries(app ${PROJECT_NAME}) 
target_link_libraries(computeDetFaces ${PROJECT_NAME})
target_link_libraries(computeDetCars ${PROJECT_NAME})
