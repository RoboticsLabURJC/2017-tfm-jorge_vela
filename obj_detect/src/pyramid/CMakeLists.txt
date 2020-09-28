
# Source files for channels features extraction from images
FILE(
  GLOB
  PYRAMID_EXTRACTION_SRC

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramid.h

  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${PYRAMID_EXTRACTION_SRC}
)

