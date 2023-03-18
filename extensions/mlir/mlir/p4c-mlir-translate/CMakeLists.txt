set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# TODO: convert the translation code into add_mlir_translation_library

add_llvm_executable(p4c-mlir-translate
    p4c-mlir-translate.cpp
    mlirgen.cpp
)

llvm_update_compile_flags(p4c-mlir-translate)

target_link_libraries(p4c-mlir-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport

  # Provides p4 parsing, ssa conversion and other AST utilities.
  p4c-mlir-lib
)

mlir_check_link_libraries(p4c-mlir-translate)
