cmake_minimum_required(VERSION 3.0.0)

project(Potato_3D_shape_OPENCV VERSION 0.1 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

link_directories(/usr/local/lib)
link_directories(/usr/lib/x86_64-linux-gnu)
find_package(LIBIGL REQUIRED QUIET)
find_package(mlpack)
find_package(Eigen3 REQUIRED)
find_package(Armadillo REQUIRED)
find_package( OpenCV REQUIRED )

include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(include)

# Add your project files
file(GLOB SRCFILES src/*.cpp)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw ${ARMADILLO_LIBRARIES} mlpack ${OpenCV_LIBS})
