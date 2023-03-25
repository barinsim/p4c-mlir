#include "ir/visitor.h"
#include "ir/ir.h"
#include "ir/pass_manager.h"
#include "ir/dump.h"

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
    } else if (p4type->is<IR::Type_Boolean>()) {
        return mlir::IntegerType::get(builder.getContext(), 1, mlir::IntegerType::Signless);
    }

    throw std::domain_error("Not implemented");
    return nullptr;
}


} // namespace


class MLIRGenImplCFG : public Inspector
{
    mlir::OpBuilder& builder;

    BasicBlock* currBlock = nullptr;

    std::unordered_map<const IR::Expression*, mlir::Value> exprToValue;

    const std::unordered_map<const BasicBlock*, mlir::Block*>& blocksMapping;

    const P4::TypeMap* typeMap = nullptr;

    // This is an internal flag that makes sure this visitor was started properly
    // using custom 'apply' method instead of the usual `node->apply(visitor)`
    bool customApplyCalled = false;

 public:
    MLIRGenImplCFG(mlir::OpBuilder &builder_,
                   const std::unordered_map<const BasicBlock *, mlir::Block *> &blocksMapping_,
                   const P4::TypeMap *typeMap_)
        : builder(builder_), blocksMapping(blocksMapping_), typeMap(typeMap_) {
            CHECK_NULL(typeMap);
        }

    // This is the main way to run this visitor, 'stmt' must be part of 'bb's components,
    // otherwise a terminator will not be able to map the successors onto MLIR blocks properly
    void apply(const IR::StatOrDecl* stmt, BasicBlock* bb) {
        CHECK_NULL(stmt, bb);
        currBlock = bb;
        customApplyCalled = true;
        stmt->apply(*this);
        customApplyCalled = false;
    }

 private:
    Visitor::profile_t init_apply(const IR::Node *node) override {
        BUG_CHECK(customApplyCalled, "Visitor was not started properly");
        return Inspector::init_apply(node);
    }

    void postorder(const IR::BoolLiteral* boolean) override {
        auto type = toMLIRType(builder, typeMap->getType(boolean));
        CHECK_NULL(type);
        mlir::Value val = builder.create<p4mlir::ConstantOp>(loc(builder, boolean), type,
                                                             (int64_t)boolean->value);
        addValue(boolean, val);
        return;
    }

    void postorder(const IR::Constant* cst) override {
        auto type = toMLIRType(builder, typeMap->getType(cst));
        CHECK_NULL(type);
        BUG_CHECK(cst->fitsInt64(), "Not implemented");
        mlir::Value val =
            builder.create<p4mlir::ConstantOp>(loc(builder, cst), type, cst->asInt64());
        addValue(cst, val);
        return;
    }

    void postorder(const IR::ReturnStatement* ret) override {
        builder.create<p4mlir::ReturnOp>(loc(builder, ret));
    }

    bool preorder(const IR::IfStatement* ifStmt) override {
        visit(ifStmt->condition);
        mlir::Value cond = toValue(ifStmt->condition);
        // TODO: remove the magic constants
        mlir::Block* tBlock = getMLIRBlock(currBlock->succs.at(0));
        mlir::Block* fBlock;

        // There are some wierd cases where if stmt can have 1 successor,
        // in that case set both targets to the same block,
        // mlir::cf::CondBranchOp can handle that
        // TODO: it stems from the canonicalization of the CFG, remove it
        if (currBlock->succs.size() == 1) {
            fBlock = tBlock;
        } else {
            fBlock = getMLIRBlock(currBlock->succs.at(1));;
        }

        // TODO: make p4.cond?
        builder.create<mlir::cf::CondBranchOp>(loc(builder, ifStmt), cond, tBlock,
                                               mlir::ValueRange(), fBlock, mlir::ValueRange());
        return false;

    }

 private:
    void addValue(const IR::Expression* expr, mlir::Value value) {
        BUG_CHECK(!exprToValue.count(expr), "Expression already has a value");
        exprToValue.insert({expr, value});
    }

    mlir::Value toValue(const IR::Expression* expr) const {
        BUG_CHECK(exprToValue.count(expr), "Could not retrive a value for the expression");
        return exprToValue.at(expr);
    }

    mlir::Block* getMLIRBlock(const BasicBlock* p4block) const {
        BUG_CHECK(blocksMapping.count(p4block), "Could not retrieve corresponding MLIR block");
        return blocksMapping.at(p4block);
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
        std::unordered_map<const BasicBlock*, mlir::Block*> mapping;
        CFGWalker::preorder(cfg.at(action), [&](BasicBlock* bb) {
            auto& block = actOp.getBody().emplaceBlock();
            mapping.insert({bb, &block});
        });

        // Fill the MLIR Blocks with Ops
        MLIRGenImplCFG cfgGen(builder, mapping, typeMap);
        CFGWalker::preorder(cfg.at(action), [&](BasicBlock* bb) {
            BUG_CHECK(mapping.count(bb), "Could not retrive MLIR block");
            auto* block = mapping.at(bb);
            builder.setInsertionPointToStart(block);
            auto& comps = bb->components;
            std::for_each(comps.begin(), comps.end(), [&](auto* stmt) {
                cfgGen.apply(stmt, bb);
            });
        });

        // Terminate blocks which had no terminator in CFG.
        // If the block has 1 successor insert BranchOp.
        // If the block has no successors insert ReturnOp.
        CFGWalker::preorder(cfg.at(action), [&](BasicBlock* bb) {
            BUG_CHECK(mapping.count(bb), "Could not retrive MLIR block");
            auto* block = mapping.at(bb);
            if (!block->empty() && block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
                return;
            }
            BUG_CHECK(bb->succs.size() <= 1, "Non-terminated block can have at most 1 successor");
            auto loc = builder.getUnknownLoc();
            if (bb->succs.size() == 1) {
                auto* succ = bb->succs.front();
                BUG_CHECK(mapping.count(succ), "Could not retrive MLIR block");
                builder.create<mlir::cf::BranchOp>(loc, mapping.at(succ), mlir::ValueRange());
            } else {
                builder.create<p4mlir::ReturnOp>(loc);
            }
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