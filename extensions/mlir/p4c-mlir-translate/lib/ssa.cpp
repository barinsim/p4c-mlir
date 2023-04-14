#include "ssa.h"
#include <variant>
#include <stack>
#include <exception>
#include <optional>
#include <iterator>

#include "ir/ir.h"
#include "ir/visitor.h"
#include "ir/dump.h"

#include "frontends/common/resolveReferences/referenceMap.h"
#include "frontends/p4/typeMap.h"

#include "lib/log.h"

#include "cfgBuilder.h"
#include "domTree.h"


namespace p4mlir {


bool isPrimitiveType(const IR::Type *type) {
    CHECK_NULL(type);
    return type->is<IR::Type::Bits>() || type->is<IR::Type::Varbits>() ||
           type->is<IR::Type::Boolean>() || type->is<IR::Type_StructLike>();
}

bool AllocateVariables::GatherAllReferencedVariables::preorder(const IR::Declaration* decl) {
    auto* type = typeMap->getType(decl, true);
    if (!isPrimitiveType(type)) {
        return true;
    }
    vars.insert(decl);
    return true;
}

bool AllocateVariables::GatherAllReferencedVariables::preorder(const IR::PathExpression* pe) {
    auto* type = typeMap->getType(pe, true);
    if (!isPrimitiveType(type)) {
        return true;
    }
    CHECK_NULL(pe->path);
    auto* decl = refMap->getDeclaration(pe->path, true);
    vars.insert(decl);
    return true;
}

ordered_set<const IR::IDeclaration*> AllocateVariables::getRegVariables() const {
    ordered_set<const IR::IDeclaration*> rv;
    std::for_each(allocation.begin(), allocation.end(), [&](auto& kv) {
        auto& [decl, allocType] = kv;
        if (allocType == AllocType::REG) {
            rv.insert(decl);
        }
    });
    return rv;
}

bool AllocateVariables::isRegVariable(const IR::IDeclaration* decl) const {
    BUG_CHECK(allocation.count(decl), "Querying unreferenced variable");
    return allocation.at(decl) == AllocType::REG;
}

Visitor::profile_t AllocateVariables::init_apply(const IR::Node* node) {
    // Gather all referenced variables and by default allocate them into SSA registers
    GatherAllReferencedVariables gather(refMap, typeMap);
    node->apply(gather);
    auto referencedVars = gather.getReferencedVars();

    std::for_each(referencedVars.begin(), referencedVars.end(), [&](auto* decl) {
        // Skip already allocated variables from previous 'apply()' calls
        if (allocation.count(decl)) {
            return;
        }
        allocation.insert({decl, AllocType::REG});
    });

    return Inspector::init_apply(node);
}

bool AllocateVariables::preorder(const IR::Parameter* param) {
    // Allocate `out` and `inout` parameters onto STACK.
    // This aligns with allocation of variables used as arguments at call sites
    if (param->direction == IR::Direction::InOut || param->direction == IR::Direction::Out) {
        allocateToStack(param);
    }
    return false;
}

bool AllocateVariables::preorder(const IR::PathExpression* pe) {
    auto* type = typeMap->getType(pe, true);
    if (!isPrimitiveType(type)) {
        return true;
    }
    CHECK_NULL(pe->path);
    auto* decl = refMap->getDeclaration(pe->path, true);

    // Variables used as an out or inout arguments must be STACK allocated.
    // This aligns with parameter allocations
    if (findContext<IR::Argument>() && isWrite()) {
        allocateToStack(decl);
    }

    // TODO: written structs_likes

    return true;
}

void AllocateVariables::allocateToStack(const IR::IDeclaration* decl) {
    BUG_CHECK(allocation.count(decl), "Allocating unreferenced variable");
    allocation[decl] = AllocType::STACK;
}

void GatherSSAReferences::Builder::addRead(const IR::PathExpression *pe,
                                           const IR::IDeclaration *decl) {
    CHECK_NULL(pe, decl);
    reads.push_back({pe, decl});
}

void GatherSSAReferences::Builder::addWrite(const IR::PathExpression *pe,
                                            const IR::IDeclaration *decl) {
    CHECK_NULL(pe, decl);
    writes.push_back({pe, decl});
}

void GatherSSAReferences::Builder::addWrite(const IR::IDeclaration* decl) {
    CHECK_NULL(decl);
    writes.push_back({decl, decl});
}

bool GatherSSAReferences::preorder(const IR::PathExpression *pe) {
    auto* type = typeMap->getType(pe);
    if (!isPrimitiveType(type)) {
        return true;
    }
    CHECK_NULL(pe->path);
    auto* decl = refMap->getDeclaration(pe->path, true);
    if (!ssaVariables.count(decl)) {
        return true;
    }
    BUG_CHECK(!(isRead() && isWrite()), "ReadWrite context cannot be expressed as an SSA form");
    if (isRead()) {
        b.addRead(pe, decl);
    }
    if (isWrite()) {
        b.addWrite(pe, decl);
    }
    return true;
}

bool GatherSSAReferences::preorder(const IR::Declaration* decl) {
    if (ssaVariables.count(decl)) {
        b.addWrite(decl);
    }
    return true;
}

bool GatherSSAReferences::preorder(const IR::IfStatement* ifStmt) {
    visit(ifStmt->condition);
    return false;
}

bool GatherSSAReferences::preorder(const IR::SwitchStatement* switchStmt) {
    (void)switchStmt;
    throw std::logic_error("Not implemented");
    return true;
}

void SSAInfo::Builder::addPhi(const BasicBlock* bb, const IR::IDeclaration* var) {
    CHECK_NULL(bb, var);
    phiInfo[bb].insert({var, Phi()});
}

void SSAInfo::Builder::numberRef(
    ID id, std::variant<const IR::IDeclaration *, const IR::PathExpression *> ref) {
    BUG_CHECK(!ssaRefIDs.count(ref), "Renumbering SSA reference");
    ssaRefIDs[ref] = id;
}

void SSAInfo::Builder::numberPhiDestination(ID id, const BasicBlock *block,
                                            const IR::IDeclaration *var) {
    BUG_CHECK(phiInfo.count(block) && phiInfo.at(block).count(var),
                "Phi node does not exist");
    Phi& phi = phiInfo.at(block).at(var);
    BUG_CHECK(!phi.destination.has_value(),
                "Phi node destination should not be numbered at this point");
    phi.destination = id;
}

void SSAInfo::Builder::numberPhiSource(ID id, const BasicBlock *block, const IR::IDeclaration *var,
                                       const BasicBlock *source) {
    BUG_CHECK(phiInfo.count(block) && phiInfo.at(block).count(var),
                "Phi node does not exist");
    BUG_CHECK(!phiInfo.at(block).at(var).sources[source].has_value(),
                "Phi node source should not be numbered at this point");
    phiInfo.at(block).at(var).sources[source] = id;
}

bool SSAInfo::Builder::phiExists(const BasicBlock* bb, const IR::IDeclaration* var) const {
    CHECK_NULL(bb, var);
    return phiInfo.count(bb) && phiInfo.at(bb).count(var);
}

ordered_set<const IR::IDeclaration *> SSAInfo::Builder::getPhiInfo(
    const BasicBlock *bb) const {
    if (!phiInfo.count(bb)) {
        return {};
    }
    ordered_set<const IR::IDeclaration*> decls;
    std::transform(phiInfo.at(bb).begin(), phiInfo.at(bb).end(),
                    std::inserter(decls, decls.end()), [](auto &p) { return p.first; });
    return decls;
}

SSAInfo::SSAInfo(const IR::IApply* context, std::pair<const IR::Node*, const BasicBlock*> cfg,
                 const P4::ReferenceMap *refMap, const P4::TypeMap *typeMap) {
    CHECK_NULL(cfg.first, cfg.second, refMap, typeMap);
    auto* entry = cfg.second;
    auto* decl = cfg.first;
    BUG_CHECK(decl->is<IR::P4Action>() || decl->is<IR::BlockStatement>(),
              "SSA can be computed only for actions and apply methods");

    Builder b;

    // Allocate variables either into SSA reg or onto a stack
    AllocateVariables alloc(refMap, typeMap);
    decl->apply(alloc);
    if (context) {
        context->getApplyParameters()->apply(alloc);
    }
    auto ssaVars = alloc.getRegVariables();

    // For each variable collect blocks where it is written
    auto collectWrites = [&]() {
        ordered_map<const IR::IDeclaration*, ordered_set<const BasicBlock*>> rv;
        CFGWalker::forEachBlock(entry, [&](auto* bb) {
            for (auto* stmt : bb->components) {
                GatherSSAReferences refs(typeMap, refMap, ssaVars);
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

    // Creates phi nodes for a variable 'var' which is written in 'writeBlocks'
    auto createPhiNodes = [&](const IR::IDeclaration* var,
                                const ordered_set<const BasicBlock* >& writeBlocks) {
        ordered_set<const BasicBlock*> visited;
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
                // If var is not in scope in the block, do not add the phi
                if (!bb->scope.isVisible(var)) {
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

    // In P4 dialect, action parameters are represented as block parameters of the entry block.
    // MLIR block parameters are equal to phi nodes which is how we represent action parameters
    // here. This applies if parameters belong directly to the body, which is the case only for P4
    // actions
    auto createActionParameters = [&]() {
        auto* action = decl->to<IR::P4Action>();
        BUG_CHECK(action, "Parameters can be added only for P4 actions");
        // Add both reg and stack variables
        auto* params = action->getParameters()->getDeclarations();
        std::for_each(params->begin(), params->end(), [&](auto* decl) {
            b.addPhi(entry, decl);
            if (alloc.isRegVariable(decl)) {
                b.numberRef((ID)0, decl);
            }
        });
    };

    // Assigns number to each SSA value reference including phi nodes.
    // Considers outter apply parameters of a context as well
    auto numberSSAValues = [&]() {
        ordered_map<const IR::IDeclaration*, ID> nextIDs;
        ordered_map<const IR::IDeclaration*, std::stack<ID>> stkIDs;
        if (context) {
            auto* params = context->getApplyParameters();
            std::for_each(params->begin(), params->end(), [&](auto* param) {
                stkIDs[param].push((ID)0);
                b.numberRef((ID)0, param);
            });
        }
        rename(entry, b, nextIDs, stkIDs, domTree, typeMap, refMap, ssaVars);
    };

    auto varToBlocks = collectWrites();
    for (auto& [var, blocks] : varToBlocks) {
        createPhiNodes(var, blocks);
    }
    if (decl->is<IR::P4Action>()) {
        createActionParameters();
    }
    numberSSAValues();

    phiInfo = b.movePhiInfo();
    ssaRefIDs = b.moveRefsInfo();
}

void SSAInfo::rename(const BasicBlock *block, Builder &b,
                     ordered_map<const IR::IDeclaration *, ID> &nextIDs,
                     ordered_map<const IR::IDeclaration *, std::stack<ID>> &stkIDs,
                     const DomTree *domTree, const P4::TypeMap *typeMap,
                     const P4::ReferenceMap *refMap,
                     const ordered_set<const IR::IDeclaration *>& ssaVars) const {
    // This is used to pop the correct number of elements from 'stkIDs'
    // once we are getting out of the recursion
    ordered_map<const IR::IDeclaration*, int> IDsAdded;

    auto vars = b.getPhiInfo(block);
    for (auto* var : vars) {
        b.numberPhiDestination(nextIDs[var], block, var);
        stkIDs[var].push(nextIDs[var]);
        ++IDsAdded[var];
        ++nextIDs[var];
    }
    for (auto* stmt : block->components) {
        GatherSSAReferences refs(typeMap, refMap, ssaVars);
        stmt->apply(refs);
        for (RefInfo& read : refs.getReads()) {
            BUG_CHECK(!stkIDs[read.decl].empty(), "Cannot number SSA use without previous def");
            b.numberRef(stkIDs[read.decl].top(), read.ref);
        }
        for (RefInfo& write : refs.getWrites()) {
            b.numberRef(nextIDs[write.decl], write.ref);
            stkIDs[write.decl].push(nextIDs[write.decl]);
            ++IDsAdded[write.decl];
            ++nextIDs[write.decl];
        }
    }
    for (auto* succ : block->succs) {
        auto succVars = b.getPhiInfo(succ);
        for (auto* var : succVars) {
            BUG_CHECK(!stkIDs[var].empty(), "Cannot number phi node argument without previous def");
            b.numberPhiSource(stkIDs[var].top(), succ, var, block);
        }
    }
    for (auto* child : domTree->children(block)) {
        rename(child, b, nextIDs, stkIDs, domTree, typeMap, refMap, ssaVars);
    }
    for (auto[var, cnt] : IDsAdded) {
        auto& stk = stkIDs.at(var);
        while (cnt--) {
            stk.pop();
        }
    }
}


} // namespace p4mlir
