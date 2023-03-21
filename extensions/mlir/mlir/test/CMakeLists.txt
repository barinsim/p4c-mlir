configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(P4C_MLIR_TEST_DEPENDS
        FileCheck count not
        p4c-mlir-translate
        )

add_lit_testsuite(check-p4c-mlir "Running the p4c-mlir regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${P4C_MLIR_TEST_DEPENDS}
        )
set_target_properties(check-p4c-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(P4C-MLIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${P4C_MLIR_TEST_DEPENDS})
