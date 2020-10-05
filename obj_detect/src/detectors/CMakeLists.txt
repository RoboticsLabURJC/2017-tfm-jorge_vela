
# Source files for channels features extraction from images
FILE(
  GLOB
  DETECTORS_EXTRACTION_SRC

  ${CMAKE_CURRENT_SOURCE_DIR}/BadacostDetector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BadacostDetector.h

  ${CMAKE_CURRENT_SOURCE_DIR}/DetectionRectangle.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DetectionRectangle.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ClassifierConfig.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ClassifierConfig.h
  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${DETECTORS_EXTRACTION_SRC}
)

