
# Source files for channels features extraction from images
FILE(
  GLOB
  DETECTORS_EXTRACTION_SRC

  ${CMAKE_CURRENT_SOURCE_DIR}/BadacostDetector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BadacostDetector.h

  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${DETECTORS_EXTRACTION_SRC}
)

