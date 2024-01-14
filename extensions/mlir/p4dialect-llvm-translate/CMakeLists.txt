set(LLVM_TARGET_DEFINITIONS include/P4DialectToLLVMPatterns.td)
mlir_tablegen(include/P4DialectToLLVMPatterns.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRP4DialectToLLVMINCGEN)

add_mlir_conversion_library(MLIRP4DialectToLLVM
        lib/P4DialectToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        include

        DEPENDS
        MLIRP4DialectToLLVMINCGEN
        intrinsics_gen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRP4Dialect
        MLIRLLVMDialect
        MLIRAnalysis
        MLIRTransformUtils
)

target_include_directories(MLIRP4DialectToLLVM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(MLIRP4DialectToLLVM PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)