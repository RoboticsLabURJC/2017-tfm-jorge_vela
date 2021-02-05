
# Source files for channels features extraction from images
FILE(
  GLOB
  PYRAMID_EXTRACTION_SRC

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramid.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/pyramid/ChannelsPyramid.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidComputeAllStrategy.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/pyramid/ChannelsPyramidComputeAllStrategy.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidComputeAllParallelStrategy.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/pyramid/ChannelsPyramidComputeAllParallelStrategy.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidApproximatedStrategy.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/pyramid/ChannelsPyramidApproximatedStrategy.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidApproximatedParallelStrategy.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/pyramid/ChannelsPyramidApproximatedParallelStrategy.h
  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${PYRAMID_EXTRACTION_SRC}
)

