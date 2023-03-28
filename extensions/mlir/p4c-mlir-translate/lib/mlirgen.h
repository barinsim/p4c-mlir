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
#include "ssa.h"

#include "P4Dialect.h"
#include "P4Ops.h"


namespace p4mlir {

using SSARefType = std::pair<std::variant<const IR::PathExpression *, const IR::IDeclaration *>,
                             p4mlir::SSAInfo::ID>;

namespace {


mlir::Location loc(mlir::OpBuilder& builder, const IR::Node* node) {
    // TODO:
    return mlir::FileLineColLoc::get(builder.getStringAttr("test/file.p4"), 42, 422);
}

mlir::Type toMLIRType(mlir::OpBuilder& builder, const IR::Type* p4type) {
    CHECK_NULL(p4type);
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

// Creates block arguments for jump from 'bb' to 'succ'.
// The order of arguments corresponds to phi arguments stored within 'ssaInfo'.
// 'ssaInfo' should be also used to create block parameters to match the order.
std::vector<mlir::Value> createBlockArgs(const SSAInfo &ssaInfo, const BasicBlock *bb,
                                         const BasicBlock *succ,
                                         const std::map<SSARefType, mlir::Value> &refMap) {
    std::vector<mlir::Value> rv;
    auto phiInfo = ssaInfo.getPhiInfo(succ);
    for (auto &[decl, phi] : phiInfo) {
        BUG_CHECK(phi.sources.count(bb), "Phi node does not contain argument for the block");
        auto id = phi.sources.at(bb).value();
        SSARefType ref{decl, id};
        BUG_CHECK(refMap.count(ref), "Could not resolve phi argument into value");
        auto argVal = refMap.at(ref);
        rv.push_back(argVal);
    }
    return rv;
}


} // namespace


class MLIRGenImplCFG : public Inspector
{
    mlir::OpBuilder& builder;

    BasicBlock* currBlock = nullptr;

    ordered_map<const IR::Expression*, mlir::Value> exprToValue;

    std::map<SSARefType, mlir::Value>& ssaRefToValue;

    const ordered_map<const BasicBlock*, mlir::Block*>& blocksMapping;

    const P4::TypeMap* typeMap = nullptr;
    const P4::ReferenceMap* refMap = nullptr;

    const SSAInfo& ssaInfo;

    // This is an internal flag that makes sure this visitor was started properly
    // using custom 'apply' method instead of the usual `node->apply(visitor)`
    bool customApplyCalled = false;

 public:
    MLIRGenImplCFG(mlir::OpBuilder &builder_,
                   const ordered_map<const BasicBlock *, mlir::Block *> &blocksMapping_,
                   const P4::TypeMap *typeMap_, const P4::ReferenceMap *refMap_,
                   const SSAInfo& ssaInfo_,
                   std::map<SSARefType, mlir::Value>& ssaRefToValue_)
        : builder(builder_),
          blocksMapping(blocksMapping_),
          typeMap(typeMap_),
          refMap(refMap_),
          ssaInfo(ssaInfo_),
          ssaRefToValue(ssaRefToValue_) {
        CHECK_NULL(typeMap, refMap);
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
    }

    void postorder(const IR::ReturnStatement* ret) override {
        builder.create<p4mlir::ReturnOp>(loc(builder, ret));
    }

    void postorder(const IR::AssignmentStatement* assign) override {
        // TODO: writes through pointer
        BUG_CHECK(assign->left->is<IR::PathExpression>(), "Not implemented");

        auto rValue = toValue(assign->right);
        mlir::Value value = builder.create<p4mlir::CopyOp>(loc(builder, assign), rValue);
        addValue(assign->left, value);
    }

    void postorder(const IR::Declaration_Variable* decl) override {
        if (!decl->initializer) {
            // TODO: undefined variable
            BUG_CHECK(false, "Not implemented");
        }
        mlir::Value init = toValue(decl->initializer);
        mlir::Value value = builder.create<p4mlir::CopyOp>(loc(builder, decl), init);
        addValue(decl, value);
    }

    void postorder(const IR::Cast* cast) override {
        CHECK_NULL(cast->destType);
        auto src = toValue(cast->expr);
        auto targetType = toMLIRType(builder, cast->destType);
        auto value = builder.create<p4mlir::CastOp>(loc(builder, cast), targetType, src);
        addValue(cast, value);
    }

    bool preorder(const IR::IfStatement* ifStmt) override {
        visit(ifStmt->condition);
        mlir::Value cond = toValue(ifStmt->condition);
        mlir::Block* tBlock = getMLIRBlock(currBlock->getTrueSuccessor());
        mlir::Block* fBlock = getMLIRBlock(currBlock->getFalseSuccessor());
        auto tArgs =
            createBlockArgs(ssaInfo, currBlock, currBlock->getTrueSuccessor(), ssaRefToValue);
        auto fArgs =
            createBlockArgs(ssaInfo, currBlock, currBlock->getFalseSuccessor(), ssaRefToValue);
        auto l = loc(builder, ifStmt);
        builder.create<mlir::cf::CondBranchOp>(l, cond, tBlock, tArgs, fBlock, fArgs);
        return false;
    }

    void postorder(const IR::Equ* eq) override {
        auto lhsValue = toValue(eq->left);
        auto rhsValue = toValue(eq->right);
        auto res =
            builder.create<CompareOp>(loc(builder, eq), CompareOpKind::eq, lhsValue, rhsValue);
        addValue(eq, res);
    }

 private:
    bool hasValue(const IR::Node* node) const {
        // TODO:
        return true;
    }

    void addValue(const IR::Node* node, mlir::Value value) {
        // Check if 'node' represents SSA value write,
        // through IR::PathExpression or IR::IDeclaration.
        // These must be stored separately to be able to resolve
        // references of these values
        auto* decl = node->to<IR::IDeclaration>();
        if (decl && ssaInfo.isSSARef(decl)) {
            SSARefType ref{decl, ssaInfo.getID(decl)};
            ssaRefToValue.insert({ref, value});
            return;
        }
        auto* pe = node->to<IR::PathExpression>();
        if (pe && ssaInfo.isSSARef(pe)) {
            CHECK_NULL(pe->path);
            auto* decl = refMap->getDeclaration(pe->path);
            SSARefType ref{decl, ssaInfo.getID(pe)};
            ssaRefToValue.insert({ref, value});
            return;
        }
        BUG_CHECK(node->is<IR::Expression>(),
                  "Value can be associated only with an expression at this point");
        exprToValue.insert({node->to<IR::Expression>(), value});
    }

    // Some SSA value references do not have corresponding AST node.
    // For example references within phi nodes.
    // These values can be retrived directly via 'decl' and its SSA 'id'
    mlir::Value toValue(const IR::IDeclaration* decl, SSAInfo::ID id) const {
        SSARefType ref{decl, id};
        BUG_CHECK(ssaRefToValue.count(ref), "No matching value found");
        return ssaRefToValue.at(ref);
    }

    mlir::Value toValue(const IR::Node* node) const {
        // SSA value reference is handled differently.
        // The mlir::Value is not connected directly to the
        // AST node, rather to unique {decl, SSA id} pair
        auto* pe = node->to<IR::PathExpression>();
        if (pe && ssaInfo.isSSARef(pe)) {
            CHECK_NULL(pe->path);
            auto* decl = refMap->getDeclaration(pe->path);
            return toValue(decl, ssaInfo.getID(pe));
        }
        BUG_CHECK(node->is<IR::Expression>(), "At this point node must be an expression");
        return exprToValue.at(node->to<IR::Expression>());
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
    const P4::ReferenceMap* refMap = nullptr;
    const CFGBuilder::CFGType& cfg;

 public:
    MLIRGenImpl(mlir::OpBuilder &builder_, const P4::TypeMap *typeMap_,
                const P4::ReferenceMap *refMap_, const CFGBuilder::CFGType &cfg_)
        : builder(builder_), typeMap(typeMap_), refMap(refMap_), cfg(cfg_) {
        CHECK_NULL(typeMap, refMap);
    }

 private:
    bool preorder(const IR::P4Action* action) override {
        // Create ActionOp
        ::llvm::StringRef name(action->getName().toString());
        auto actOp = builder.create<p4mlir::ActionOp>(loc(builder, action), name);
        auto parent = builder.getBlock();

        // For each BasicBlock create MLIR Block and insert it into the ActionOp region
        BUG_CHECK(cfg.count(action), "Could not retrive cfg");
        ordered_map<const BasicBlock*, mlir::Block*> mapping;
        CFGWalker::preorder(cfg.at(action), [&](BasicBlock* bb) {
            auto& block = actOp.getBody().emplaceBlock();
            mapping.insert({bb, &block});
        });

        // Create ssa mapping for this action
        auto cfgIt = cfg.find(action);
        SSAInfo ssaInfo(*cfgIt, refMap, typeMap);

        // Stores mapping of P4 ssa values to its mlir counterparts.
        // This mapping is used during MLIRgen to resolve references
        std::map<SSARefType, mlir::Value> ssaRefMap;

        // For each mlir::Block insert block arguments using phi nodes info.
        // These block arguments must be bound to {decl, ssa id} pairs,
        // so that references can be resolved during MLIRgen
        CFGWalker::preorder(cfg.at(action), [&](BasicBlock* bb) {
            mlir::Block* block = mapping.at(bb);
            auto phiInfo = ssaInfo.getPhiInfo(bb);
            for (auto& [decl, phi] : phiInfo) {
                auto loc = builder.getUnknownLoc();
                auto type = toMLIRType(builder, typeMap->getType(decl->to<IR::Declaration>()));
                mlir::BlockArgument arg = block->addArgument(type, loc);
                // Bind the arg
                auto id = phi.destination.value();
                SSARefType ref{decl, id};
                BUG_CHECK(!ssaRefMap.count(ref), "Binding already exists");
                ssaRefMap.insert({ref, arg});
            }
        });

        // Fill the MLIR Blocks with Ops
        MLIRGenImplCFG cfgGen(builder, mapping, typeMap, refMap, ssaInfo, ssaRefMap);
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
            builder.setInsertionPointToEnd(block);
            auto loc = builder.getUnknownLoc();
            if (bb->succs.size() == 1) {
                auto* succ = bb->succs.front();
                BUG_CHECK(mapping.count(succ), "Could not retrive MLIR block");
                auto args = createBlockArgs(ssaInfo, bb, succ, ssaRefMap);
                builder.create<mlir::cf::BranchOp>(loc, mapping.at(succ), args);
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
        passes.push_back(new MLIRGenImpl(builder, typeMap, refMap, *cfg));
    }
};


mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const IR::P4Program *program);


} // namespace p4mlir