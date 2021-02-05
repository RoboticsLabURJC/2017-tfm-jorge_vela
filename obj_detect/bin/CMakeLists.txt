
# ----------------- testProgram -----------------------------------------
add_executable(app testProgram.cpp)
target_include_directories(
        app
        PUBLIC
        "${PROJECT_SOURCE_DIR}/obj_detect/include"
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
  "${PROJECT_SOURCE_DIR}/obj_detect/include"
)
target_link_libraries(computeDetFaces ${PROJECT_NAME})

# ------------------ computeDetectionCars -----------------------------
add_executable(computeDetCars computeDetectionCars.cpp)
target_include_directories(computeDetCars
  PUBLIC
  "${PROJECT_SOURCE_DIR}/obj_detect/include"
)
target_link_libraries(computeDetCars ${PROJECT_NAME})
