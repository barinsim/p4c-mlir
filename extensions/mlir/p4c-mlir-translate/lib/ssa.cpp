#include "ssa.h"

#include <exception>
#include <iterator>
#include <optional>
#include <stack>
#include <variant>

#include "cfg.h"
#include "domTree.h"
#include "frontends/common/resolveReferences/referenceMap.h"
#include "frontends/p4/typeMap.h"
#include "ir/dump.h"
#include "ir/ir.h"
#include "ir/visitor.h"
#include "lib/log.h"

namespace p4mlir {


bool isPrimitiveType(const IR::Type *type) {
    CHECK_NULL(type);
    return type->is<IR::Type::Bits>() || type->is<IR::Type::Varbits>() ||
           type->is<IR::Type::Boolean>() || type->is<IR::Type_StructLike>() ||
           type->is<IR::Type_BaseList>();
}

bool GatherAllocatableVariables::preorder(const IR::Declaration_Instance* decl) {
    vars.insert(decl);
    return true;
}

bool GatherAllocatableVariables::preorder(const IR::Declaration_Variable *decl) {
    vars.insert(decl);
    return true;
}

bool GatherAllocatableVariables::preorder(const IR::Declaration_Constant *decl) {
    vars.insert(decl);
    return true;
}

bool GatherAllocatableVariables::preorder(const IR::P4Table *decl) {
    vars.insert(decl);
    return true;
}

bool GatherAllocatableVariables::preorder(const IR::Parameter *param) {
    // Parameters of extern methods/functions are not used within a P4 program, no need to allocate
    if (findContext<IR::Type_Method>() || findContext<IR::Method>()) {
        return false;
    }
    vars.insert(param);
    return true;
}

void Allocation::setAllocatableVariables(const ordered_set<const IR::IDeclaration*>& allocatable) {
    std::for_each(allocatable.begin(), allocatable.end(), [&](const IR::IDeclaration *decl) {
        BUG_CHECK(!data.count(decl), "Duplicate variable");
        data[decl] = AllocType::REG;
    });
}

void Allocation::set(const IR::IDeclaration* decl, AllocType allocType) {
    BUG_CHECK(data.count(decl), "Could not find allocation");
    data[decl] = allocType;
}

AllocType Allocation::get(const IR::IDeclaration* decl) const {
    BUG_CHECK(data.count(decl), "Could not find allocation");
    return data.at(decl);
}

ordered_set<const IR::IDeclaration*> Allocation::getAllOf(AllocType type) const {
    ordered_set<const IR::IDeclaration*> rv;
    std::for_each(data.begin(), data.end(), [&](auto& kv) {
        auto* decl = kv.first;
        auto allocType = kv.second;
        if (allocType == type) {
            rv.insert(decl);
        }
    });
    return rv;
}

Visitor::profile_t AllocateVariables::init_apply(const IR::Node* node) {
    // Gather all allocatable variables
    GatherAllocatableVariables gather(refMap, typeMap);
    node->apply(gather);

    // By default allocate all variables to REG, variables are then demoted to
    // other allocations, depending on the context in which they are used
    allocation.setAllocatableVariables(gather.getReferencedVars());

    return Inspector::init_apply(node);
}

void AllocateVariables::end_apply(const IR::Node *node) {
    GatherAllocatableVariables gather(refMap, typeMap);
    node->apply(gather);
    auto vars = gather.getReferencedVars();

    // Sanity check the allocation
    std::for_each(vars.begin(), vars.end(), [&](const IR::IDeclaration* decl) {
        auto* type = typeMap->getType(decl->to<IR::Declaration>(), true);
        AllocType allocType = allocation.get(decl);

        if (type->is<IR::Type_Bits>() || type->is<IR::Type_StructLike>() ||
            type->is<IR::Type_Boolean>() || type->is<IR::Type_BaseList>()) {
            BUG_CHECK(allocType == AllocType::REG ||
                      allocType == AllocType::STACK ||
                      allocType == AllocType::CONSTANT_MEMBER,
                      "Unexpected allocation type");
        }
        else if (type->is<IR::Type_Extern>() || type->is<IR::Type_SpecializedCanonical>() ||
                 type->is<IR::Type_Specialized>() || type->is<IR::Type_Control>() ||
                 type->is<IR::Type_Parser>() || type->is<IR::P4Control>() ||
                 type->is<IR::P4Parser>()) {
            BUG_CHECK(allocType == AllocType::EXTERN ||
                      allocType == AllocType::EXTERN_MEMBER ||
                      allocType == AllocType::EXTERN,
                      "Unexpected allocation type");
            if (allocType == AllocType::EXTERN) {
                BUG_CHECK(decl->is<IR::Parameter>(), "Expected parameter");
            }
        }
        else if (type->is<IR::Type_Table>()) {
            BUG_CHECK(allocType == AllocType::EXTERN_MEMBER, "Unexpected allocation type");
        }
        else {
            BUG_CHECK(false, "Unknown type");
        }
    });
}

bool AllocateVariables::preorder(const IR::Parameter* param) {
    // Parameters of externs are not used within a p4 program, no need to allocate
    if (findContext<IR::Type_Method>() || findContext<IR::Method>()) {
        return false;
    }

    // Allocate extern parameters as EXTERN
    auto* type = typeMap->getType(param, true);
    if (type->is<IR::Type_Extern>() || type->is<IR::Type_SpecializedCanonical>() ||
        type->is<IR::Type_Specialized>() || type->is<IR::Type_Control>() ||
        type->is<IR::Type_Parser>() || type->is<IR::P4Control>() || type->is<IR::P4Parser>()) {
        allocation.set(param, AllocType::EXTERN);
        return false;
    }

    // Allocate `out` and `inout` parameters onto STACK.
    if (param->direction == IR::Direction::InOut || param->direction == IR::Direction::Out) {
        allocation.set(param, AllocType::STACK);
        return false;
    }

    return false;
}

bool AllocateVariables::preorder(const IR::P4Control* control) {
    // Allocate out-of-apply local declarations to STACK.
    // This is overly conservative, but simplifies many things, like referencing these variables
    // within table declarations, since we do not have to compute SSA numbering for them
    auto& decls = control->controlLocals;
    std::for_each(decls.begin(), decls.end(), [&](const IR::IDeclaration* decl) {
        if (decl->is<IR::Declaration_Variable>()) {
            allocation.set(decl, AllocType::STACK);
        }
    });
    return true;
}

bool AllocateVariables::preorder(const IR::P4Parser* parser) {
    // Allocate out-of-state local declarations to STACK.
    // This is overly conservative, but simplifies many things, like referencing these variables
    // within parser states, since we do not have to compute SSA numbering for them
    auto& decls = parser->parserLocals;
    std::for_each(decls.begin(), decls.end(), [&](const IR::IDeclaration* decl) {
        if (decl->is<IR::Declaration_Variable>()) {
            allocation.set(decl, AllocType::STACK);
        }
    });
    return true;
}

bool AllocateVariables::preorder(const IR::P4Table* table) {
    allocation.set(table, AllocType::EXTERN_MEMBER);
}

bool AllocateVariables::preorder(const IR::PathExpression* pe) {
    // Skip references that do not reference allocatable types
    auto* type = typeMap->getType(pe, true);
    if (type->is<IR::Type_MethodBase>() || type->is<IR::Type_State>()) {
        return true;
    }

    // Skip externally allocated objects, those are allocated while vising its declarations
    if (type->is<IR::Type_Extern>() || type->is<IR::Type_SpecializedCanonical>() ||
        type->is<IR::Type_Specialized>() || type->is<IR::Type_Control>() ||
        type->is<IR::Type_Parser>() || type->is<IR::P4Control>() || type->is<IR::P4Parser>() ||
        type->is<IR::Type_Table>()) {
        return true;
    }

    // Get the referenced declaration
    CHECK_NULL(pe->path);
    auto* decl = refMap->getDeclaration(pe->path, true);

    // Variables used as out or inout arguments must be STACK allocated.
    // This aligns with parameter allocations
    if (findContext<IR::Argument>() && isWrite()) {
        allocation.set(decl, AllocType::STACK);
    }

    // Written composite type variables must be allocated to stack
    if (type->is<IR::Type_StructLike>() && isWrite()) {
        allocation.set(decl, AllocType::STACK);
    }

    return true;
}

bool AllocateVariables::preorder(const IR::Declaration_Instance* decl) {
    auto* context = findContext<IR::IContainer>();
    if (!context) {
        BUG_CHECK(false, "Not implemented");
    }
    allocation.set(decl, AllocType::EXTERN_MEMBER);
}

bool AllocateVariables::preorder(const IR::Declaration_Constant* decl) {
    auto* context = findContext<IR::IContainer>();
    if (!context) {
        BUG_CHECK(false, "Not implemented");
    }
    allocation.set(decl, AllocType::CONSTANT_MEMBER);
}

void GatherSSAReferences::addRead(const IR::PathExpression *pe,
                                           const IR::IDeclaration *decl) {
    CHECK_NULL(pe, decl);
    reads.push_back({pe, decl});
}

void GatherSSAReferences::addWrite(const IR::PathExpression *pe,
                                            const IR::IDeclaration *decl) {
    CHECK_NULL(pe, decl);
    writes.push_back({pe, decl});
}

void GatherSSAReferences::addWrite(const IR::IDeclaration* decl) {
    CHECK_NULL(decl);
    writes.push_back({decl, decl});
}

bool GatherSSAReferences::preorder(const IR::PathExpression *pe) {
    auto* type = typeMap->getType(pe, true);
    if (!isPrimitiveType(type)) {
        return true;
    }
    CHECK_NULL(pe->path);
    auto* decl = refMap->getDeclaration(pe->path, true);
    if (allocation.get(decl) != AllocType::REG) {
        return true;
    }
    BUG_CHECK(!(isRead() && isWrite()), "ReadWrite context cannot be expressed as an SSA var");
    if (isRead()) {
        addRead(pe, decl);
    }
    if (isWrite()) {
        addWrite(pe, decl);
    }
    return true;
}

bool GatherSSAReferences::preorder(const IR::Declaration* decl) {
    auto* type = typeMap->getType(decl, true);
    if (isPrimitiveType(type) && allocation.get(decl) == AllocType::REG) {
        addWrite(decl);
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

void SSAInfo::addPhi(const BasicBlock* bb, const IR::IDeclaration* var) {
    CHECK_NULL(bb, var);
    phiInfo[bb].insert({var, PhiInfo()});
}

void SSAInfo::numberRef(
    ID id, std::variant<const IR::IDeclaration *, const IR::PathExpression *> ref) {
    BUG_CHECK(!ssaRefIDs.count(ref), "Renumbering SSA reference");
    ssaRefIDs[ref] = id;
}

void SSAInfo::numberPhiDestination(ID id, const BasicBlock *block, const IR::IDeclaration *var) {
    BUG_CHECK(phiInfo.count(block) && phiInfo.at(block).count(var),
                "Phi node does not exist");
    PhiInfo& phi = phiInfo.at(block).at(var);
    BUG_CHECK(!phi.destination.has_value(),
                "Phi node destination should not be numbered at this point");
    phi.destination = id;
}

void SSAInfo::numberPhiSource(ID id, const BasicBlock *block, const IR::IDeclaration *var,
                                       const BasicBlock *source) {
    BUG_CHECK(phiInfo.count(block) && phiInfo.at(block).count(var),
                "Phi node does not exist");
    BUG_CHECK(!phiInfo.at(block).at(var).sources[source].has_value(),
                "Phi node source should not be numbered at this point");
    phiInfo.at(block).at(var).sources[source] = id;
}

bool SSAInfo::phiExists(const BasicBlock* bb, const IR::IDeclaration* var) const {
    CHECK_NULL(bb, var);
    return phiInfo.count(bb) && phiInfo.at(bb).count(var);
}

ID SSAInfo::getID(std::variant<const IR::IDeclaration *, const IR::PathExpression *> ref) const {
    BUG_CHECK(ssaRefIDs.count(ref), "Querying non-existent SSA numbering");
    return ssaRefIDs.at(ref);
}

ordered_map<const IR::IDeclaration*, PhiInfo> SSAInfo::getPhiInfo(const BasicBlock *bb) const {
    if (phiInfo.count(bb)) {
        return phiInfo.at(bb);
    }
    return {};
}

bool MakeSSAInfo::preorder(const IR::P4Control* control) {
    // Collect REG parameters
    auto* applyParams = control->getApplyParameters();
    auto* ctrParams = control->getConstructorParameters();
    ordered_set<const IR::IDeclaration*> regParams;
    for (auto* params : {applyParams, ctrParams}) {
        std::for_each(params->begin(), params->end(), [&](const IR::Parameter* param) {
            if (allocation.get(param) == AllocType::REG) {
                regParams.insert(param);
            }
        });
    }

    // Number REG parameters.
    // Those can be only read, which means they can be simply assigned 0
    std::for_each(regParams.begin(), regParams.end(), [&](const IR::IDeclaration* param) {
        ssaInfo.numberRef((ID)0, param);
    });

    // Number out-of-apply declarations + apply method
    traverseCFG(regParams, cfgInfo.get(control));

    return true;
}

bool MakeSSAInfo::preorder(const IR::P4Parser* parser) {
    // Collect REG parameters
    auto* applyParams = parser->getApplyParameters();
    auto* ctrParams = parser->getConstructorParameters();
    ordered_set<const IR::IDeclaration*> regParams;
    for (auto* params : {applyParams, ctrParams}) {
        std::for_each(params->begin(), params->end(), [&](const IR::Parameter* param) {
            if (allocation.get(param) == AllocType::REG) {
                regParams.insert(param);
            }
        });
    }

    // Number REG parameters
    // Those can be only read, which means they can be simply assigned 0
    std::for_each(regParams.begin(), regParams.end(), [&](const IR::IDeclaration* param) {
        ssaInfo.numberRef((ID)0, param);
    });

    // Number out-of-state local declarations
    traverseCFG(regParams, cfgInfo.get(parser));

    return true;
}

bool MakeSSAInfo::preorder(const IR::P4Action* action) {
    // Number action parameters, those are either STACK allocated or REG allocated but only
    // read. Which means we can simply assign 0
    auto* params = action->parameters;
    std::for_each(params->begin(), params->end(), [&](const IR::Parameter* param) {
        // Skip non-REG allocated vars
        if (allocation.get(param) != AllocType::REG) {
            return;
        }
        auto dir = param->direction;
        BUG_CHECK(dir == IR::Direction::None || dir == IR::Direction::In,
                  "Unexpected param direction");
        ssaInfo.numberRef((ID)0, param);
    });

    // Collect REG parameters.
    // Those include ctr/apply params and action params
    ordered_set<const IR::IDeclaration*> regParams;
    std::for_each(params->begin(), params->end(), [&](const IR::Parameter* param) {
        if (allocation.get(param) == AllocType::REG) {
            regParams.insert(param);
        }
    });
    if (auto* control = findContext<IR::P4Control>()) {
        // Collect ctr/apply params
        auto* applyParams = control->getApplyParameters();
        auto* ctrParams = control->getConstructorParameters();
        for (auto* params : {applyParams, ctrParams}) {
            std::for_each(params->begin(), params->end(), [&](const IR::Parameter* param) {
                if (allocation.get(param) == AllocType::REG) {
                    regParams.insert(param);
                }
            });
        }
    }

    // Number action body
    traverseCFG(regParams, cfgInfo.get(action));

    return true;
}

bool MakeSSAInfo::preorder(const IR::ParserState* state) {
    // Collect REG apply/ctr parameters of the enclosing block
    auto* parser = findContext<IR::P4Parser>();
    auto* applyParams = parser->getApplyParameters();
    auto* ctrParams = parser->getConstructorParameters();
    ordered_set<const IR::IDeclaration*> regParams;
    for (auto* params : {applyParams, ctrParams}) {
        std::for_each(params->begin(), params->end(), [&](const IR::Parameter* param) {
            if (allocation.get(param) == AllocType::REG) {
                regParams.insert(param);
            }
        });
    }

    // Number parser state body
    traverseCFG(regParams, cfgInfo.get(state));

    return true;
}

bool MakeSSAInfo::preorder(const IR::PathExpression* pe) {
    // REG references within table declarations can either be constructor params or IN-direction
    // apply params of the enclosing P4Control. None of those variables can be written within the
    // control block. This means the references can be simply assigned the same SSA number which was
    // assigned to the declaration. This applies because the out-of-apply local declarations are
    // currently getting STACK allocation
    auto* type = typeMap->getType(pe, true);
    if (findContext<IR::P4Table>() && isPrimitiveType(type)) {
        CHECK_NULL(pe->path);
        auto* decl = refMap->getDeclaration(pe->path, true);
        if (allocation.get(decl) != AllocType::REG) {
            return true;
        }

        // Check the assumptions under which this SSA numbering algorithm works
        auto* control = findContext<IR::P4Control>();
        CHECK_NULL(control);
        auto& applyParams = control->getApplyParameters()->parameters;
        auto& ctrParams = control->getConstructorParameters()->parameters;
        std::vector<const IR::Parameter*> allParams;
        std::copy(applyParams.begin(), applyParams.end(), std::back_inserter(allParams));
        std::copy(ctrParams.begin(), ctrParams.end(), std::back_inserter(allParams));
        bool isParam = std::any_of(allParams.begin(), allParams.end(),
                                   [&](const IR::Parameter *param) { return param == decl; });
        BUG_CHECK(isParam, "Reference of a P4Control apply/ctr parameter expected");

        // Retrieve and assign the SSA numbering that was assigned to the declaration
        ID ssaID = ssaInfo.getID(decl);
        ssaInfo.numberRef(ssaID, pe);
    }

    return true;
}

bool MakeSSAInfo::preorder(const IR::Declaration_Instance* decl) {
    if (decl->initializer || !decl->properties.empty()) {
        BUG_CHECK(false, "Not implemented");
    }
    // Number REG variables used as costructor arguments, those can be only constructor parameters
    // of the enclosing control/parser
    GatherSSAReferences refs(typeMap, refMap, allocation);
    decl->apply(refs);
    BUG_CHECK(refs.getWrites().empty(), "Unexpected SSA variable write");
    auto reads = refs.getReads();
    std::for_each(reads.begin(), reads.end(), [&](RefInfo info) {
        ID ssaID = ssaInfo.getID(info.decl);
        ssaInfo.numberRef(ssaID, info.ref);
    });
    return false;
}

bool MakeSSAInfo::preorder(const IR::Declaration_Constant* decl) {
    // Number REG variables used within initializer, those can be only constructor parameters
    // of the enclosing control/parser
    GatherSSAReferences refs(typeMap, refMap, allocation);
    decl->apply(refs);
    BUG_CHECK(refs.getWrites().empty(), "Unexpected SSA variable write");
    auto reads = refs.getReads();
    std::for_each(reads.begin(), reads.end(), [&](RefInfo info) {
        ID ssaID = ssaInfo.getID(info.decl);
        ssaInfo.numberRef(ssaID, info.ref);
    });
    return false;
}

void MakeSSAInfo::traverseCFG(const ordered_set<const IR::IDeclaration*>& params, CFG cfg) {
    auto collectWrites = [&]() {
        ordered_map<const IR::IDeclaration*, ordered_set<const BasicBlock*>> rv;
        CFGWalker::forEachBlock(cfg.getEntry(), [&](const BasicBlock* bb) {
            for (auto* stmt : bb->components) {
                // Skip declarations of non-primitive types, those are handled separately
                if (stmt->is<IR::Declaration>() && !stmt->is<IR::Declaration_Variable>()) {
                    continue;
                }
                GatherSSAReferences refs(typeMap, refMap, allocation);
                stmt->apply(refs);
                auto writes = refs.getWrites();
                std::for_each(writes.begin(), writes.end(), [&](auto& w) {
                    rv[w.decl].insert(bb);
                });
            }
        });
        return rv;
    };

    // Create dominator tree for this cfg
    DomTree* domTree = DomTree::fromEntryBlock(cfg.getEntry());

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
                if (ssaInfo.phiExists(bb, var)) {
                    continue;
                }
                // If var is not in scope in the block, do not add the phi
                if (!bb->scope.isVisible(var)) {
                    continue;
                }
                ssaInfo.addPhi(bb, var);
                if (!visited.count(bb)) {
                    visited.insert(bb);
                    worklist.push(bb);
                }
            }
        }
    };

    // Assigns number to each SSA value reference including phi nodes.
    // Considers parameters as well
    auto numberSSAValues = [&]() {
        ordered_map<const IR::IDeclaration*, ID> nextIDs;
        ordered_map<const IR::IDeclaration*, std::stack<ID>> stkIDs;
        // Init the stacks for the parameters, which should be already numbered
        std::for_each(params.begin(), params.end(), [&](const IR::IDeclaration* param) {
            CHECK_NULL(param);
            BUG_CHECK(!stkIDs.count(param), "Duplicate parameter");
            BUG_CHECK(allocation.get(param) == AllocType::REG, "Parameter must be REG allocated");
            auto id = ssaInfo.getID(param);
            stkIDs[param].push(id);
        });
        rename(cfg.getEntry(), nextIDs, stkIDs, domTree);
    };

    auto varToBlocks = collectWrites();
    for (auto& [var, blocks] : varToBlocks) {
        createPhiNodes(var, blocks);
    }
    numberSSAValues();
}

void MakeSSAInfo::rename(const BasicBlock *block,
                     ordered_map<const IR::IDeclaration *, ID> &nextIDs,
                     ordered_map<const IR::IDeclaration *, std::stack<ID>> &stkIDs,
                     const DomTree *domTree) const {
    // This is used to pop the correct number of elements from 'stkIDs'
    // once we are getting out of the recursion
    ordered_map<const IR::IDeclaration*, int> IDsAdded;

    auto vars = ssaInfo.getPhiInfo(block);
    for (auto& [var, _] : vars) {
        ssaInfo.numberPhiDestination(nextIDs[var], block, var);
        stkIDs[var].push(nextIDs[var]);
        ++IDsAdded[var];
        ++nextIDs[var];
    }
    for (auto* stmt : block->components) {
        GatherSSAReferences refs(typeMap, refMap, allocation);
        stmt->apply(refs);
        for (RefInfo& read : refs.getReads()) {
            BUG_CHECK(!stkIDs[read.decl].empty(), "Cannot number SSA use without previous def");
            ssaInfo.numberRef(stkIDs[read.decl].top(), read.ref);
        }
        for (RefInfo& write : refs.getWrites()) {
            ssaInfo.numberRef(nextIDs[write.decl], write.ref);
            stkIDs[write.decl].push(nextIDs[write.decl]);
            ++IDsAdded[write.decl];
            ++nextIDs[write.decl];
        }
    }
    for (auto* succ : block->succs) {
        auto succVars = ssaInfo.getPhiInfo(succ);
        for (auto& [var, _] : succVars) {
            BUG_CHECK(!stkIDs[var].empty(), "Cannot number phi node argument without previous def");
            ssaInfo.numberPhiSource(stkIDs[var].top(), succ, var, block);
        }
    }
    for (auto* child : domTree->children(block)) {
        rename(child, nextIDs, stkIDs, domTree);
    }
    for (auto[var, cnt] : IDsAdded) {
        auto& stk = stkIDs.at(var);
        while (cnt--) {
            stk.pop();
        }
    }
}


} // namespace p4mlir
