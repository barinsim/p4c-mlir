add_mlir_dialect_library(MLIRP4Dialect
        P4Dialect.cpp
        P4Ops.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../include

        DEPENDS
        MLIRP4OpsIncGen

	    LINK_LIBS PUBLIC
	    MLIRIR
        MLIRInferTypeOpInterface
)
