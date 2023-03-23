#include "ir/visitor.h"
#include "ir/ir.h"
#include "ir/pass_manager.h"

#include "lib/ordered_map.h"

#include "frontends/p4/typeMap.h"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/typeChecking/typeChecker.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Block.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "cfgBuilder.h"

#include "P4Dialect.h"
#include "P4Ops.h"


namespace p4mlir {

namespace {


mlir::Location loc(mlir::OpBuilder& builder, const IR::Node* node) {
    // TODO:
    return mlir::FileLineColLoc::get(builder.getStringAttr("test/file.p4"), 42, 422);
}

mlir::Type toMLIRType(mlir::OpBuilder& builder, const IR::Type* p4type) {
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


} // namespace


class MLIRGenImplCFG : public Inspector
{
    mlir::OpBuilder& builder;

    const P4::TypeMap* typeMap = nullptr;

 public:
    MLIRGenImplCFG(mlir::OpBuilder &builder_, const P4::TypeMap *typeMap_)
        : builder(builder_), typeMap(typeMap_) {}

 private:
    void postorder(const IR::Constant* cst) override {
        auto type = toMLIRType(builder, typeMap->getType(cst));
        CHECK_NULL(type);
        BUG_CHECK(cst->fitsInt64(), "Not implemented");
        builder.create<p4mlir::ConstantOp>(loc(builder, cst), type, cst->asInt64());
        return;
    }

    void postorder(const IR::ReturnStatement* ret) override {
        builder.create<p4mlir::ReturnOp>(loc(builder, ret));
    }
};


class MLIRGenImpl : public Inspector
{
    mlir::OpBuilder& builder;

    const P4::TypeMap* typeMap = nullptr;
    const CFGBuilder::CFGType& cfg;

 public:
    MLIRGenImpl(mlir::OpBuilder &builder_, const P4::TypeMap *typeMap_, const CFGBuilder::CFGType& cfg_)
        : builder(builder_), typeMap(typeMap_), cfg(cfg_) {}

 private:
    bool preorder(const IR::P4Action* action) override {
        // Create ActionOp
        ::llvm::StringRef name(action->getName().toString());
        auto actOp = builder.create<p4mlir::ActionOp>(loc(builder, action), name);
        auto parent = builder.getBlock();

        // For each BasicBlock create MLIR Block and insert it into the ActionOp region
        BUG_CHECK(cfg.count(action), "Could not retrive cfg");
        ordered_map<BasicBlock*, mlir::Block*> mapping;
        CFGWalker::preorder(cfg.at(action), [&](BasicBlock* bb) {
            auto& block = actOp.getBody().emplaceBlock();
            mapping.insert({bb, &block});
        });

        // Fill the MLIR Blocks with Ops
        MLIRGenImplCFG cfgGen(builder, typeMap);
        CFGWalker::preorder(cfg.at(action), [&](BasicBlock* bb) {
            BUG_CHECK(mapping.count(bb), "Could not retrive MLIR block");
            auto* block = mapping.at(bb);
            builder.setInsertionPointToStart(block);
            auto& comps = bb->components;
            std::for_each(comps.begin(), comps.end(), [&](auto* stmt) {
                stmt->apply(cfgGen);
            });
        });

        builder.setInsertionPointToEnd(parent);
        return false;
    }
};


class MLIRGen : public PassManager
{
 public:
    MLIRGen(mlir::OpBuilder& builder) {
        auto* refMap = new P4::ReferenceMap();
        auto* typeMap = new P4::TypeMap();
        auto* cfg = new CFGBuilder::CFGType();
        passes.push_back(new P4::ResolveReferences(refMap));
        passes.push_back(new P4::TypeInference(refMap, typeMap, false, true));
        passes.push_back(new P4::TypeChecking(refMap, typeMap, true));
        passes.push_back(new CFGBuilder(*cfg));
        passes.push_back(new MLIRGenImpl(builder, typeMap, *cfg));
    }
};


mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const IR::P4Program *program);


} // namespace p4mlir