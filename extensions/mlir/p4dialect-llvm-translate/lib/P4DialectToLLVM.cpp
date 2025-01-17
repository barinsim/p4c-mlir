#include "P4DialectToLLVM.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "P4Ops.h"
#include "P4Dialect.h"
#include "P4DialectToLLVMPatterns.h"

namespace {
struct P4DialectToLLVMPass
    : public PassWrapper<P4DialectToLLVMPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(P4DialectToLLVMPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect, p4mlir::P4Dialect>();
    }
    void runOnOperation() final;
};

struct P4DialectToLLVMTarget : public ConversionTarget {
    P4DialectToLLVMTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
        addLegalDialect<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect>();
    }
};
} // namespace

void P4DialectToLLVMPass::runOnOperation() {
    P4DialectToLLVMTarget target(getContext());

    RewritePatternSet patterns(&getContext());
    P4DialectToLLVMTypeConverter typeConvertor(&getContext());

    populateWithGenerated(patterns);
    patterns.add<ConstantOpPattern>(typeConvertor, &getContext());
    patterns.add<AddOpPattern>(typeConvertor, &getContext());


    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> createP4DialectToLLVMPass() {
    return std::make_unique<P4DialectToLLVMPass>();
}





