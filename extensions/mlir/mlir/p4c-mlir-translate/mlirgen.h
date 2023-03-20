#include "ir/visitor.h"
#include "ir/ir.h"
#include "ir/pass_manager.h"

#include "frontends/p4/typeMap.h"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/typeChecking/typeChecker.h"

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


class MLIRGenImpl : public Inspector
{
    mlir::OpBuilder& builder;

    const P4::TypeMap* typeMap = nullptr;

 public:
    MLIRGenImpl(mlir::OpBuilder &builder_, const P4::TypeMap *typeMap_)
        : builder(builder_), typeMap(typeMap_) {}

 private:
    bool preorder(const IR::P4Action* action) override {
        // builder.create<p4mlir::ConstantOp>(loc(action), 42);
        return true;
    }

    void postorder(const IR::Constant* cst) override {
        auto type = toMLIRType(typeMap->getType(cst));
        CHECK_NULL(type);
        BUG_CHECK(cst->fitsInt64(), "Not implemented");
        builder.create<p4mlir::ConstantOp>(loc(cst), type, cst->asInt64());
        return;
    }

 private:
    mlir::Location loc(const IR::Node* node) {
        // TODO:
        return mlir::FileLineColLoc::get(builder.getStringAttr("test/file.p4"), 42, 422);
    }

    mlir::Type toMLIRType(const IR::Type* p4type) const {
        if (p4type->is<IR::Type_InfInt>()) {
            // TODO: create special type
            return mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Signed);
        } else if (auto* bits = p4type->to<IR::Type_Bits>()) {
            auto sign = bits->isSigned ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
            int size = bits->size;
            return mlir::IntegerType::get(builder.getContext(), size, sign);
        }
        throw std::domain_error("Not implemented");
        return nullptr;
    }


};


class MLIRGen : public PassManager
{
 public:
    MLIRGen(mlir::OpBuilder& builder) {
        auto* refMap = new P4::ReferenceMap();
        auto* typeMap = new P4::TypeMap();
        passes.push_back(new P4::ResolveReferences(refMap));
        passes.push_back(new P4::TypeInference(refMap, typeMap, false, true));
        passes.push_back(new P4::TypeChecking(refMap, typeMap, true));
        passes.push_back(new MLIRGenImpl(builder, typeMap));
    }
};


mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const IR::P4Program *program);


} // namespace p4mlir