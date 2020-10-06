
cmake_minimum_required(VERSION 3.2)

project(objdetect)
add_subdirectory(obj_detect)

include_directories(
  obj_detect/include
  )

add_executable(app testProgram.cpp)

target_include_directories(app PUBLIC
  obj_detect/include 
)

target_link_libraries(app ${PROJECT_NAME}) 