#ifndef P4_DIALECT
#define P4_DIALECT

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
    let dependentDialects = ["::mlir::cf::ControlFlowDialect"];

    //let useDefaultTypePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
    let useFoldAPI = kEmitFoldAdaptorFolder;
}

//===----------------------------------------------------------------------===//
// Base p4 operation definition.
//===----------------------------------------------------------------------===//

class P4Op<string mnemonic, list<Trait> traits = []> :
        Op<P4Dialect, mnemonic, traits>;

#endif  // P4_DIALECT
