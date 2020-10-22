
# Source files for channels features extraction from images
FILE(
  GLOB
  CHANNELS_EXTRACTION_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUV.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUV.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHist.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHist.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHistPDollar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHistPDollar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHistOpenCV.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHistOpenCV.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMag.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMag.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMagPDollar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMagPDollar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMagOpenCV.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMagsOpenCV.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorACF.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorACF.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLDCF.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLDCF.h

  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.h

  ${CMAKE_CURRENT_SOURCE_DIR}/badacostDetector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/badacostDetector.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsPyramid.h

  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${CHANNELS_EXTRACTION_SRC}
)
