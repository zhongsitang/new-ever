cmake_minimum_required(VERSION 3.18)

# Test utilities
add_library(test_utils INTERFACE)
target_include_directories(test_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# C++ unit tests
add_executable(test_types test_types.cpp)
target_link_libraries(test_types PRIVATE gaussian_rt_core test_utils)

add_executable(test_device test_device.cpp)
target_link_libraries(test_device PRIVATE gaussian_rt_core test_utils)

add_executable(test_primitives test_primitives.cpp)
target_link_libraries(test_primitives PRIVATE gaussian_rt_core test_utils)

add_executable(test_accel test_accel.cpp)
target_link_libraries(test_accel PRIVATE gaussian_rt_core test_utils)

# Add tests to CTest
add_test(NAME test_types COMMAND test_types)
add_test(NAME test_device COMMAND test_device)
add_test(NAME test_primitives COMMAND test_primitives)
add_test(NAME test_accel COMMAND test_accel)

# Python integration test (optional)
if(Python_FOUND)
    add_test(
        NAME test_python_integration
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_integration.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()
