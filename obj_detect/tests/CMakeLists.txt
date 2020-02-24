#
# @file    CMakeLists.txt
# @brief   Configuration for the project
# @author  Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date    2019/07/09
#

# Name for the regression tests executable
set(
        TESTS_REGRESSION_NAME
        "test_regression"
)

# Entry point for all tests
set(
        TEST_SHARED_ENTRY_POINT
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# Sources for regression tests builds
file(
        GLOB
        TEST_REGRESSION_SHARED_SRC
        ${TEST_SHARED_ENTRY_POINT}
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# All regression tests for the library
add_executable(
        ${TESTS_REGRESSION_NAME}
        ${TEST_REGRESSION_SHARED_SRC}        
)

# Main project headers
unset(${PROJECT_NAME}_INCLUDE_DIRS)
get_target_property(
        ${PROJECT_NAME}_INCLUDE_DIRS
        ${PROJECT_NAME} INCLUDE_DIRECTORIES
)

# tests headers
target_include_directories(
        ${TESTS_REGRESSION_NAME}
        PRIVATE
        ${${PROJECT_NAME}_INCLUDE_DIRS}
)

# Create test executables
target_link_libraries(
        ${TESTS_REGRESSION_NAME}
        ${PROJECT_NAME}
        ${GTEST_BOTH_LIBRARIES}
        ${OpenCV_LIBS}
)

file(COPY ${PROJECT_SOURCE_DIR}/tests/index.jpeg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tests/index2.jpeg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tests/index3.jpeg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${PROJECT_SOURCE_DIR}/tests/luv_matlab_index.jpeg.yml  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tests/MRealImage.yaml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tests/ORealImage.yaml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${PROJECT_SOURCE_DIR}/tests/index2.jpeg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_test(
        NAME ${TESTS_REGRESSION_NAME}
        COMMAND ${PROJECT_BINARY_DIR}/tests/${TESTS_REGRESSION_NAME}
)


