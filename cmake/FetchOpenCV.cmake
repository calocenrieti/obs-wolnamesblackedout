# Local OpenCV reference (version 4.13.0)
set(OPENCV_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../opencv-4.13.0_build_for_obs/install")

find_path(OPENCV_INCLUDE_DIRS opencv2/opencv.hpp
    HINTS ${OPENCV_ROOT_DIR}/include)

if(MSVC)
  add_library(OpenCV INTERFACE)
  target_link_libraries(OpenCV INTERFACE 
      ${OPENCV_ROOT_DIR}/x64/vc17/staticlib/opencv_world4130.lib
      ${OPENCV_ROOT_DIR}/x64/vc17/staticlib/ipphal.lib
      ${OPENCV_ROOT_DIR}/x64/vc17/staticlib/ippicvmt.lib
      ${OPENCV_ROOT_DIR}/x64/vc17/staticlib/ippiw.lib
      ${OPENCV_ROOT_DIR}/x64/vc17/staticlib/ittnotify.lib
      ${OPENCV_ROOT_DIR}/x64/vc17/staticlib/zlib.lib)
  target_include_directories(OpenCV SYSTEM INTERFACE ${OPENCV_INCLUDE_DIRS})
else()
  find_library(OPENCV_LIBS opencv_world4130
      HINTS ${OPENCV_ROOT_DIR}/x64/vc17/lib)
  add_library(OpenCV INTERFACE)
  target_link_libraries(OpenCV INTERFACE ${OPENCV_LIBS})
  target_include_directories(OpenCV SYSTEM INTERFACE ${OPENCV_INCLUDE_DIRS})
endif()

if(NOT OPENCV_INCLUDE_DIRS)
    message(FATAL_ERROR "OpenCV include directories not found at ${OPENCV_ROOT_DIR}")
endif()
