cmake_minimum_required( VERSION 3.10 )
project(blind_sandbox)

list(APPEND CMAKE_PREFIX_PATH "/home/joonho/oldws/install") #!

find_package(raisim CONFIG REQUIRED)
find_package(raisimOgre 0.5.0 CONFIG REQUIRED)
find_package(TensorFlow CONFIG REQUIRED)

if (raisimOgre_FOUND)
    message(STATUS "Found raisimOgre!")
endif()

if (TensorFlow_FOUND)
    message(STATUS "Found Tensorflow!")
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIRS})
    list(APPEND ${PROJECT_NAME}_LIBRARIES ${TensorFlow_LIBRARIES})
endif()

include_directories(include
        ${PROJECT_NAME}_INCLUDE_DIRS)

add_executable(main
        applications/main.cpp)
target_link_libraries(main
        ${${PROJECT_NAME}_LIBRARIES} raisim::raisim raisim::raisimOgre)

# EOF
