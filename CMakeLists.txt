cmake_minimum_required(VERSION 3.16.3)

project(nl_icp)
set(CMAKE_CXX_STANDARD 17)

find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_library(pt2ptres src/PointToPointResidual.cpp)
target_link_libraries(pt2ptres ${CERES_LIBRARIES})

add_library(pt2plres src/PointToPlaneResidual.cpp)
target_link_libraries(pt2plres ${CERES_LIBRARIES})

add_executable(nl_icp src/nl_icp.cpp)
target_link_libraries(nl_icp pt2ptres pt2plres ${CERES_LIBRARIES} ${PCL_LIBRARIES})
