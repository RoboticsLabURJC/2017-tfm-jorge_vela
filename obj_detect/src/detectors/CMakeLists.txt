
# Source files for channels features extraction from images
FILE(
  GLOB
  DETECTORS_EXTRACTION_SRC

  ${CMAKE_CURRENT_SOURCE_DIR}/BadacostDetector.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/detectors/BadacostDetector.h

  ${CMAKE_CURRENT_SOURCE_DIR}/DetectionRectangle.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/detectors/DetectionRectangle.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ClassifierConfig.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/detectors/ClassifierConfig.h
  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${DETECTORS_EXTRACTION_SRC}
)

