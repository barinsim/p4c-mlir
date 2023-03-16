set(
    P4C_MLIR_SOURCES
    cfgBuilder.cpp
    domTree.cpp
    ssa.cpp
)

add_library(p4c-mlir-lib ${P4C_MLIR_SOURCES})
target_link_libraries (p4c-mlir-lib ${P4C_LIBRARIES} ${P4C_LIB_DEPS})
add_dependencies(p4c-mlir-lib genIR)
target_include_directories(p4c-mlir-lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(p4c-mlir main.cpp)
target_link_libraries (p4c-mlir p4c-mlir-lib)


set (
    GTEST_MLIR_SOURCES
    tests/cfgBuilder.cpp
    tests/domTree.cpp
    tests/common.cpp
    tests/ssa.cpp

    ${P4C_SOURCE_DIR}/test/gtest/helpers.cpp
    ${P4C_SOURCE_DIR}/test/gtest/gtestp4c.cpp
)

add_executable(p4c-mlir-gtest ${GTEST_MLIR_SOURCES})
target_link_libraries (p4c-mlir-gtest ${P4C_LIBRARIES} ${P4C_LIB_DEPS} p4c-mlir-lib gtest)

add_subdirectory(mlir)
