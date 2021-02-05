
# Source files for channels features extraction from images
FILE(
  GLOB
  CHANNELS_EXTRACTION_SRC
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorLUV.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUV.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorLUVPDollar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUVPDollar.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorLUVOpenCV.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUVOpenCV.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorLUVOpenCL.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLUVOpenCL.cpp


  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorGradHist.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHist.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHistPDollar.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorGradHistPDollar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHistOpenCV.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorGradHistOpenCV.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradHistOpenCL.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorGradHistOpenCL.h


  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorGradMag.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMag.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMagPDollar.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorGradMagPDollar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMagOpenCV.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorGradMagsOpenCV.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorGradMagOpenCL.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorGradMagOpenCL.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorACF.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorACF.h

  ${CMAKE_CURRENT_SOURCE_DIR}/ChannelsExtractorLDCF.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/ChannelsExtractorLDCF.h

  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  ${PROJECT_SOURCE_DIR}/obj_detect/include/channels/Utils.h
  )

# Set the sources for the main target
target_sources(
        ${PROJECT_NAME}
        PRIVATE
        ${CHANNELS_EXTRACTION_SRC}
)
