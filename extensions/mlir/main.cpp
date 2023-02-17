#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "frontends/common/parseInput.h"
#include "frontends/common/parser_options.h"
#include "frontends/p4/frontend.h"
#include "ir/ir.h"
#include "lib/gc.h"
#include "lib/log.h"

#include "cfgBuilder.h"


class BMV2Options : public CompilerOptions {
 public:
    // file to output to
    cstring outputFile = nullptr;
    
    BMV2Options() {
        registerOption(
            "-o", "outfile",
            [this](const char *arg) {
                outputFile = arg;
                return true;
            },
            "Write output to outfile");
    }
};


int main(int argc, char *const argv[]) {
    setup_gc_logging();

    AutoCompileContext autoDpdkContext(new P4CContextWithOptions<BMV2Options>);
    auto &options = P4CContextWithOptions<BMV2Options>::get().options();
    options.langVersion = CompilerOptions::FrontendVersion::P4_16;
    options.compilerVersion = "test mlir version string";
    if (options.process(argc, argv) != nullptr) {
        options.setInputFile();
    }

    auto hook = options.getDebugHook();
    const IR::P4Program *program = nullptr;
    program = P4::parseP4File(options);
    if (program == nullptr || ::errorCount() > 0) {
        return 1;
    }
    std::cout << "AFTER PARSE" << std::endl;
    dump(program);
    std::cout << std::endl << std::endl << std::endl << std::endl;
    
    try {
        P4::FrontEnd frontend;
        frontend.addDebugHook(hook);
        program = frontend.run(options, program);
    } catch (const std::exception &bug) {
        std::cerr << bug.what() << std::endl;
        return 1;
    }

    std::cout << "AFTER FRONTEND" << std::endl;
    dump(program);

    auto cfgBuilder = new p4mlir::CFGBuilder;
    program->apply(*cfgBuilder);

    for (auto& [decl, bb] : cfgBuilder->getCFG()) {
        std::cout << decl->getName() << '\n';
        std::cout << toString(bb, 1) << '\n' << '\n';
    }

    return errorCount() != 0;
}
