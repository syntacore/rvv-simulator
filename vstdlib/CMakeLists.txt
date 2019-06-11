cmake_minimum_required(VERSION 3.0)

project(vstdlib) 

add_library(vstdlib STATIC
    vstdlib.hpp
    memcpy.cpp
    memset.cpp
    memmove.cpp
    memcpy_backward.cpp
)
set_target_properties(vstdlib PROPERTIES FOLDER "Libs")
target_include_directories(vstdlib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(vstdlib PUBLIC riscv32)

target_compile_features(vstdlib PUBLIC ${CXX_FEATURES})
