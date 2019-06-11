cmake_minimum_required(VERSION 3.0)

project(Vendetta_tests)

set(tests
    unit_tests
    stdlib
    vector_examples
)

foreach(name ${tests})
    add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp")
    set_target_properties(${name} PROPERTIES FOLDER "tests")
    add_test(test_${name} ${name})
endforeach()

target_link_libraries(unit_tests riscv32)
target_link_libraries(stdlib riscv32 vstdlib)
target_link_libraries(vector_examples riscv32)
