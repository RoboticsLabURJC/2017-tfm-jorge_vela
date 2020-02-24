
# Source files for channels features extraction from images
FILE(
  GLOB
  CHANNELS_EXTRACTION_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUV.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUV.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHist.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHist.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMag.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMag.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ImageResample.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ImageResample.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsCompute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsCompute.h
  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${CHANNELS_EXTRACTION_SRC}
)
