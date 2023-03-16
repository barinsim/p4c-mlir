set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(p4c-mlir-translate
    p4c-mlir-translate.cpp
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

  # Provides p4 parsing and ssa conversion.
  # SYSTEM kw is used to suppress warnings from p4c headers.
  SYSTEM
  p4c-mlir-lib
)

mlir_check_link_libraries(p4c-mlir-translate)
