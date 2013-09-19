cmake_minimum_required(VERSION 2.8)
project(loopclosuredetector)

cmake_policy(SET CMP0015 NEW)

set(CMAKE_BUILD_TYPE "Debug")

message("Project ${PROJECT_NAME} build Type: ${CMAKE_BUILD_TYPE}")

find_package( OpenCV REQUIRED )

add_definitions(-std=c++11)

include_directories(${OpenCV_INCLUDE_DIR})
link_directories ( ${OpenCV_LIBS})

find_package( PCL REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

find_package (Threads REQUIRED)

link_directories(/usr/local/lib)

add_executable(lcd_vector BowMatchers/bowextractor.cpp MOSAIC/pclvisualizerthread.cpp MOSAIC/tools.cpp MOSAIC/mosaic.cpp MOSAIC/Triangulation/singlecameratriangulator.cpp MOSAIC/Triangulation/normaloptimizer.cpp MOSAIC/Triangulation/neighborhoodsgenerator.cpp MOSAIC/DescriptorsMatcher/descriptorsmatcher.cpp BowMatchers/vectorbowmatcher.cpp  BowMatchers/bowmatcher.cpp matchviewer.cpp framepose.cpp frameposelist.cpp main_vector.cpp)
add_executable(lcd_precomputed BowMatchers/bowextractor.cpp MOSAIC/pclvisualizerthread.cpp MOSAIC/tools.cpp MOSAIC/mosaic.cpp MOSAIC/Triangulation/singlecameratriangulator.cpp MOSAIC/Triangulation/normaloptimizer.cpp MOSAIC/Triangulation/neighborhoodsgenerator.cpp MOSAIC/DescriptorsMatcher/descriptorsmatcher.cpp BowMatchers/vectorbowmatcher.cpp  BowMatchers/bowmatcher.cpp matchviewer.cpp framepose.cpp frameposelist.cpp main_precomputed.cpp)
#add_executable(couples BowMatchers/bowextractor.cpp MOSAIC/pclvisualizerthread.cpp MOSAIC/tools.cpp MOSAIC/mosaic.cpp MOSAIC/Triangulation/singlecameratriangulator.cpp MOSAIC/Triangulation/normaloptimizer.cpp MOSAIC/Triangulation/neighborhoodsgenerator.cpp MOSAIC/DescriptorsMatcher/descriptorsmatcher.cpp BowMatchers/vectorbowmatcher.cpp  BowMatchers/bowmatcher.cpp matchviewer.cpp framepose.cpp frameposelist.cpp main_couple_selector.cpp)

target_link_libraries(lcd_vector ${OpenCV_LIBS} ${PCL_LIBRARIES} lmfit )
target_link_libraries(lcd_precomputed ${OpenCV_LIBS} ${PCL_LIBRARIES} lmfit )
#target_link_libraries(couples ${OpenCV_LIBS} ${PCL_LIBRARIES} lmfit )

install(TARGETS lcd_vector RUNTIME DESTINATION bin)
install(TARGETS lcd_precomputed RUNTIME DESTINATION bin)
#install(TARGETS couples RUNTIME DESTINATION bin)

add_subdirectory(BowMatchers)
add_subdirectory(MOSAIC)