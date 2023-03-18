#include "frontends/common/parser_options.h"
#include "frontends/common/options.h"
#include "frontends/common/parseInput.h"

#include "mlirgen.h"


const IR::P4Program* parseP4(int argc, char** argv) {
    AutoCompileContext acc(new P4CContextWithOptions<CompilerOptions>);

    auto& context = P4CContextWithOptions<CompilerOptions>::get();
    auto& options = context.options();

    options.langVersion = CompilerOptions::FrontendVersion::P4_16;
    options.compilerVersion = "test mlir version string";
    if (options.process(argc, argv) != nullptr) {
        options.setInputFile();
    }

    auto* program = P4::parseP4File(options);

    // ::errorCount() must be called before the compile context is popped
    if (!program || ::errorCount() > 0) {
        return nullptr;
    }
    return program;
}


int main(int argc, char** argv) {
    const IR::P4Program* program = parseP4(argc, argv);
    if (!program) {
        return 1;
    }

    mlir::MLIRContext context;
    context.getOrLoadDialect<p4mlir::P4Dialect>();
    auto moduleOp = p4mlir::mlirGen(context, program);
    if (!moduleOp) {
        return 1;
    }

    moduleOp->dump();
    return 0;
}