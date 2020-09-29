
# Source files for channels features extraction from images
FILE(
  GLOB
  PYRAMID_EXTRACTION_SRC

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramid.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidComputeAllStrategy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidComputeAllStrategy.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidComputeAllParallelStrategy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidComputeAllParallelStrategy.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidApproximatedStrategy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramidApproximated`Strategy.h

  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${PYRAMID_EXTRACTION_SRC}
)

