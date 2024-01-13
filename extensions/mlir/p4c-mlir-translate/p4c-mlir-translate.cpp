#include "lib/gc.h"

#include "frontends/common/parser_options.h"
#include "frontends/common/options.h"
#include "frontends/common/parseInput.h"
#include "frontends/p4/toP4/toP4.h"

#include "mlir/Pass/PassManager.h"

#include "P4DialectToLLVM.h"

#include "mlirgen.h"


const IR::P4Program* parseP4(int argc, char** argv) {
    auto& context = P4CContextWithOptions<CompilerOptions>::get();
    auto& options = context.options();

    options.langVersion = CompilerOptions::FrontendVersion::P4_16;
    options.compilerVersion = "test mlir version string";
    if (options.process(argc, argv) != nullptr) {
        options.setInputFile();
    }

    auto* program = P4::parseP4File(options);
    if (!program || ::errorCount() > 0) {
        return nullptr;
    }

//    P4::FrontEnd frontend;
//    program = frontend.run(options, program);
//    if (!program || ::errorCount() > 0) {
//        return nullptr;
//    }

    return program;
}

void runLLVMConversion(mlir::ModuleOp moduleOp) {
    mlir::PassManager pm(moduleOp->getName());
    pm.addPass(createP4DialectToLLVMPass());
    auto result = pm.run(moduleOp);
    if (mlir::failed(result)) {
        llvm::errs() << "conversion to llvm failed";
    }
}


int main(int argc, char** argv) {
    AutoCompileContext acc(new P4CContextWithOptions<CompilerOptions>);

    bool dumpAST = false;
    if (argc > 2 && std::string(argv[2]) == "--emit-ast") {
        dumpAST = true;
        --argc;
    }

    const IR::P4Program* program = parseP4(argc, argv);
    if (!program) {
        return 1;
    }

    if (dumpAST) {
        dump(program);
        P4::dumpP4(program);
        return 0;
    }

    // MLIR uses thread local storage which is not registered by GC,
    // which causes double deletes
    GC_disable();

    mlir::MLIRContext context;
    context.getOrLoadDialect<p4mlir::P4Dialect>();

    auto moduleOp = p4mlir::mlirGen(context, program);
    if (!moduleOp) {
        return 1;
    }

    runLLVMConversion(moduleOp.get());

    moduleOp->print(llvm::outs());

    return 0;
}