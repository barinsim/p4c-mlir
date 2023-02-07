#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "control-plane/bfruntime_ext.h"
#include "control-plane/p4RuntimeSerializer.h"
#include "frontends/common/applyOptionsPragmas.h"
#include "frontends/common/parseInput.h"
#include "frontends/common/parser_options.h"
#include "frontends/p4/frontend.h"
#include "ir/ir.h"
#include "ir/json_loader.h"
#include "lib/error.h"
#include "lib/exceptions.h"
#include "lib/exename.h"
#include "lib/gc.h"
#include "lib/log.h"


int main(int argc, char *const argv[]) {
    setup_gc_logging();

    AutoCompileContext autoDpdkContext(new DPDK::DpdkContext);
    auto &options = DPDK::DpdkContext::get().options();
    options.langVersion = CompilerOptions::FrontendVersion::P4_16;
    options.compilerVersion = DPDK_VERSION_STRING;
    if (options.process(argc, argv) != nullptr) {
        if (options.loadIRFromJson == false) options.setInputFile();
    }

    auto hook = options.getDebugHook();
    const IR::P4Program *program = nullptr;
    const IR::ToplevelBlock *toplevel = nullptr;
    program = P4::parseP4File(options);
    if (program == nullptr || ::errorCount() > 0) {
        return 1;
    }
    
    try {
        P4::FrontEnd frontend;
        frontend.addDebugHook(hook);
        program = frontend.run(options, program);
    } catch (const std::exception &bug) {
        std::cerr << bug.what() << std::endl;
        return 1;
    }

    return errorCount() != 0;
}
