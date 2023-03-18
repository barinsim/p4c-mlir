#ifndef STANDALONE_DIALECT
#define STANDALONE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def P4Dialect : Dialect {
    let name = "p4";
    let summary = "A p4 out-of-tree MLIR dialect.";
    let description = [{Test p4 mlir dialect description}];
    let cppNamespace = "p4mlir";

    let useFoldAPI = kEmitFoldAdaptorFolder;
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class P4Op<string mnemonic, list<Trait> traits = []> :
        Op<P4Dialect, mnemonic, traits>;

#endif // STANDALONE_DIALECT

 // ../../../../../llvm-project/build/bin/mlir-tblgen -gen-dialect-decls P4Dialect.td -I../../../../../llvm-project/mlir/include/