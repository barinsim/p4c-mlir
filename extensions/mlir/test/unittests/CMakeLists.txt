set (
    GTEST_MLIR_SOURCES
    cfgBuilderTests.cpp
    domTreeTests.cpp
    ssaTests.cpp
    common.cpp

    ${P4C_SOURCE_DIR}/test/gtest/helpers.cpp
    ${P4C_SOURCE_DIR}/test/gtest/gtestp4c.cpp
)

add_executable(p4c-mlir-gtest ${GTEST_MLIR_SOURCES})
target_link_libraries(p4c-mlir-gtest p4c-mlir-translate-lib gtest)

