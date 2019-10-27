
# Source files for channels features extraction from images
FILE(
  GLOB
  CHANNELS_EXTRACTION_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUV.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUV.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorMex.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorMex.h
  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${CHANNELS_EXTRACTION_SRC}
)
