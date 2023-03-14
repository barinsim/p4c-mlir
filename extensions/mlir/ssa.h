#ifndef BACKENDS_MLIR_SSA_H_
#define BACKENDS_MLIR_SSA_H_


#include <variant>
#include <unordered_set>
#include <unordered_map>
#include <stack>
#include <exception>
#include "ir/ir.h"
#include "ir/visitor.h"
#include "ir/dump.h"
#include "frontends/common/resolveReferences/referenceMap.h"
#include "frontends/p4/typeMap.h"
#include "lib/log.h"
#include "cfgBuilder.h"
#include "domTree.h"


namespace p4mlir {


namespace {

bool isPrimitiveType(const IR::Type *type) {
    CHECK_NULL(type);
    return type->is<IR::Type::Bits>() || type->is<IR::Type::Varbits>() ||
           type->is<IR::Type::Boolean>();
}

}


class GatherOutArgsScalars : public Inspector, P4WriteContext
{
    class Builder {
        std::unordered_set<const IR::IDeclaration*> decls;
     public:
        void add(const IR::IDeclaration* decl) { decls.insert(decl); }
        std::unordered_set<const IR::IDeclaration*> get() const { return decls; }
    };

    Builder b;

    const P4::ReferenceMap* refMap;
    const P4::TypeMap* typeMap;

 public:
    GatherOutArgsScalars(const P4::ReferenceMap *refMap_, const P4::TypeMap *typeMap_)
        : refMap(refMap_), typeMap(typeMap_) {
        CHECK_NULL(refMap_);
        CHECK_NULL(typeMap_);
    }

    std::unordered_set<const IR::IDeclaration*> get() const {
        return b.get();
    }

 private:
    bool preorder(const IR::PathExpression* pe) override {
        if (!isWrite() || !findContext<IR::Argument>()) {
            return true;
        }
        auto* type = typeMap->getType(pe);
        if (isPrimitiveType(type)) {
            CHECK_NULL(pe);
            b.add(refMap->getDeclaration(pe->path));
        }
        return true;
    }
};


struct RefInfo {
    std::variant<const IR::IDeclaration*, const IR::PathExpression*> ref;
    // decl == ref for declarations
    const IR::IDeclaration* decl;
};


class GatherSSAReferences : public Inspector, P4WriteContext
{
    class Builder {
        std::vector<RefInfo> reads;
        std::vector<RefInfo> writes;
     public:
        void addRead(const IR::PathExpression* pe, const IR::IDeclaration* decl) {
            CHECK_NULL(pe, decl);
            reads.push_back({pe, decl});
        }
        void addWrite(const IR::PathExpression* pe, const IR::IDeclaration* decl) {
            CHECK_NULL(pe, decl);
            writes.push_back({pe, decl});
        }
        void addWrite(const IR::IDeclaration* decl) {
             CHECK_NULL(decl);
             writes.push_back({decl, decl});
        }
        std::vector<RefInfo> getReads() const { return reads; };
        std::vector<RefInfo> getWrites() const { return writes; };
    };

    Builder b;

    const P4::TypeMap* typeMap;
    const P4::ReferenceMap* refMap;

    // Output of GatherOutArgsScalars
    const std::unordered_set<const IR::IDeclaration*> forbidden;

public:
   GatherSSAReferences(const P4::TypeMap *typeMap_, const P4::ReferenceMap *refMap_,
                       std::unordered_set<const IR::IDeclaration *> forbidden_)
       : typeMap(typeMap_), refMap(refMap_), forbidden(std::move(forbidden_)) {}

   std::vector<RefInfo> getReads() const { return b.getReads(); }
   std::vector<RefInfo> getWrites() const { return b.getWrites(); }

private:
   bool preorder(const IR::PathExpression *pe) override {
        auto* type = typeMap->getType(pe);
        if (!isPrimitiveType(type)) {
            return true;
        }
        CHECK_NULL(pe->path);
        auto* decl = refMap->getDeclaration(pe->path);
        if (forbidden.count(decl)) {
            return true;
        }
        BUG_CHECK(!(isRead() && isWrite()), "ReadWrite context cannot be expressed as a SSA form");
        if (isRead()) {
            b.addRead(pe, decl);
        }
        if (isWrite()) {
            b.addWrite(pe, decl);
        }
        return true;
    }

    bool preorder(const IR::Declaration* decl) override {
        b.addWrite(decl);
        return true;
    }

    bool preorder(const IR::IfStatement* ifStmt) override {
        visit(ifStmt->condition);
        return false;
    }

    bool preorder(const IR::SwitchStatement* switchStmt) override {
        (void)switchStmt;
        throw std::logic_error("Not implemented");
        return true;
    }
};


class SSAInfo
{
    using ID = std::size_t;

    struct Phi {
        std::optional<ID> destination;
        std::unordered_map<const BasicBlock*, std::optional<ID>> sources;
    };

    // For each basic block stores its phi nodes.
    // Each phi node belongs to a variable (IR::Declaration).
    // Phi node for var V looks like this:
    //      V = phi(V, ..., V)
    // 1 argument for each predecessor.
    std::unordered_map<const BasicBlock*, std::unordered_map<const IR::IDeclaration*, Phi>>
        phiInfo;

    class Builder {
        decltype(SSAInfo::phiInfo) phiInfo;

     public:
        void addPhi(const BasicBlock* bb, const IR::IDeclaration* var) {
            CHECK_NULL(bb, var);
            phiInfo[bb].insert({var, Phi()});
        }
        bool phiExists(const BasicBlock* bb, const IR::IDeclaration* var) const {
            CHECK_NULL(bb, var);
            return phiInfo.count(bb) && phiInfo.at(bb).count(var);
        }
        decltype(phiInfo) movePhiInfo() const {
            return std::move(phiInfo);
        }
    };

    Builder b;

public:
    std::unordered_map<const IR::IDeclaration*, Phi> getPhiInfo(const BasicBlock* bb) const {
        if (phiInfo.count(bb)) {
            return phiInfo.at(bb);
        }
        return {};
    }

    SSAInfo(std::pair<const IR::IDeclaration*, const BasicBlock*> cfg,
            const P4::ReferenceMap* refMap, const P4::TypeMap* typeMap) {
        CHECK_NULL(cfg.first, cfg.second, refMap, typeMap);
        auto* entry = cfg.second;
        auto* func = cfg.first->to<IR::Declaration>();
        // TODO: pass Declaration instead of the cast
        BUG_CHECK(func, "");

        // Collect variables that cannot be stored into SSA values
        GatherOutArgsScalars g(refMap, typeMap);
        func->to<IR::Declaration>()->apply(g);
        auto forbidden = g.get();

        // For each variable collect blocks where it is written
        auto collectWrites = [&]() {
            std::unordered_map<const IR::IDeclaration*, std::unordered_set<const BasicBlock*>> rv;
            CFGWalker::forEachBlock(entry, [&](auto* bb) {
                for (auto* stmt : bb->components) {
                    GatherSSAReferences refs(typeMap, refMap, forbidden);
                    stmt->apply(refs);
                    auto writes = refs.getWrites();
                    std::for_each(writes.begin(), writes.end(), [&](auto& w) {
                        rv[w.decl].insert(bb);
                    });
                }
            });
            return rv;
        };

        DomTree* domTree = DomTree::fromEntryBlock(entry);

        // Creats phi nodes for a variable 'var' which is written in 'writeBlocks'
        auto createPhiNodes = [&](const IR::IDeclaration* var,
                                  const std::unordered_set<const BasicBlock* >& writeBlocks) {
            std::unordered_set<const BasicBlock*> visited;
            std::stack<const BasicBlock*> worklist;
            std::for_each(writeBlocks.begin(), writeBlocks.end(), [&](auto *bb) {
                worklist.push(bb);
                visited.insert(bb);
            });
            while (!worklist.empty()) {
                auto* curr = worklist.top();
                worklist.pop();
                auto domFrontier = domTree->domFrontier(curr);
                for (auto* bb : domFrontier) {
                    if (b.phiExists(bb, var)) {
                        continue;
                    }
                    b.addPhi(bb, var);
                    if (!visited.count(bb)) {
                        visited.insert(bb);
                        worklist.push(bb);
                    }
                }
            }
        };

        auto declToBlocks = collectWrites();
        for (auto& [decl, blocks] : declToBlocks) {
            createPhiNodes(decl, blocks);
        }

        phiInfo = b.movePhiInfo();
    }
};


} // namespace p4mlir


#endif /* BACKENDS_MLIR_SSA_H_ */