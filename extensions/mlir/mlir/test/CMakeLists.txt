configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-p4c-mlir "Running the p4c-mlir tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS p4c-mlir-gtest
        ARGS --show-all
        )
set_target_properties(check-p4c-mlir PROPERTIES FOLDER "Tests")

file(COPY ${CMAKE_SOURCE_DIR}/p4include DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/unittests)

add_subdirectory(unittests)