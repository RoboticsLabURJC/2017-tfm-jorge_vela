
# Source files for channels features extraction from images
FILE(
  GLOB
  CHANNELS_EXTRACTION_SRC

  ${CMAKE_CURRENT_SOURCE_DIR}/badacostDetector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/badacostDetector.h

  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${CHANNELS_EXTRACTION_SRC}
)

