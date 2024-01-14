#ifndef P4C_P4DIALECTTOLLVMPATTERNS_H
#define P4C_P4DIALECTTOLLVMPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

#include "P4DialectToLLVMHelpers.h"
#include "P4DialectToLLVMPatterns.h.inc"


class P4DialectToLLVMTypeConverter : public TypeConverter {
 public:
    P4DialectToLLVMTypeConverter(MLIRContext* ctx) {
        auto castFromSignless = [](OpBuilder &builder, Type type,
                                    ValueRange inputs, Location loc) -> std::optional<Value> {
            if (inputs.size() != 1 || !inputs[0].getType().isa<mlir::IntegerType>()) {
                return std::nullopt;
            }
            auto cast = builder.create<p4mlir::CastOp>(loc, type, inputs);
            return cast;
        };
        addSourceMaterialization(castFromSignless);

        addConversion([ctx](mlir::IntegerType type) {
            if (type.isSignless()) {
                return type;
            }
            unsigned width = type.getWidth();
            return mlir::IntegerType::get(ctx, width, mlir::IntegerType::Signless);
        });
    }
};

/**
 * p4.constant 10 : si16 -> arith.constant 10 : i16
 */
class ConstantOpPattern : public OpConversionPattern<p4mlir::ConstantOp> {
 public:
    using OpConversionPattern<p4mlir::ConstantOp>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(p4mlir::ConstantOp op, p4mlir::ConstantOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto attr = op.getValue().dyn_cast<mlir::IntegerAttr>();
        auto type = attr.getType().dyn_cast<mlir::IntegerType>();

        if (!attr || !type) {
            return failure();
        }

        auto cstOp = rewriter.create<mlir::arith::ConstantOp>(
            op.getLoc(), createSignlessAttr(rewriter, attr), createSignlessType(rewriter, type));

        rewriter.replaceOp(op, {cstOp});

        return success();
    }
};

/**
 * p4.add si8 si8 -> llvm.add i8 i8
 */
class AddOpPattern : public OpConversionPattern<p4mlir::AddOp> {
 public:
    using OpConversionPattern<p4mlir::AddOp>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(p4mlir::AddOp op, p4mlir::AddOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto addOp =
            rewriter.create<mlir::LLVM::AddOp>(op->getLoc(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(op, {addOp});
        return success();
    }
};

#endif  // P4C_P4DIALECTTOLLVMPATTERNS_H
