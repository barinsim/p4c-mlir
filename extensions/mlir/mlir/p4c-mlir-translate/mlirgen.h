#include "ir/visitor.h"
#include "ir/ir.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "P4Dialect.h"
#include "P4Ops.h"


namespace p4mlir {


class MLIRGen : public Inspector
{
    mlir::OpBuilder& builder;

 public:
    MLIRGen(mlir::OpBuilder& builder_) : builder(builder_) {

    }

 private:
    bool preorder(const IR::P4Action* action) override {
        builder.create<p4mlir::ConstantOp>(loc(builder), 42);
        return true;
    }

    mlir::Location loc(mlir::OpBuilder &builder) {
        return mlir::FileLineColLoc::get(builder.getStringAttr("test/file.p4"), 42, 422);
    }
};


mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const IR::P4Program *program);


} // namespace p4mlir