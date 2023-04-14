#ifndef BACKENDS_MLIR_TESTS_COMMON_H_
#define BACKENDS_MLIR_TESTS_COMMON_H_

#include <string>

#include "cfgBuilder.h"
#include "lib/ordered_map.h"
#include "lib/ordered_set.h"

#include "frontends/common/resolveReferences/referenceMap.h"
#include "frontends/p4/typeMap.h"

namespace p4mlir::tests {


BasicBlock* getByName(const std::map<const IR::Node*, BasicBlock*>&, const std::string&);

// This is a simple way how to get 'BasicBlock' that contains 'stmt' statement.
// It relies on a unique string representation of the statement within the whole program.
BasicBlock* getByStmtString(BasicBlock*, const std::string&);

// Converts container of IR::IDeclaration to unordered set of names of these declarations.
// Names withing 'decls' must be unique.
template <typename T>
std::unordered_set<cstring> names(T decls) {
    std::unordered_set<cstring> res;
    for (auto* d : decls) {
        const auto& name = d->getName().name;
        if (res.count(name)) {
            throw std::domain_error("Declaration names must be unique");
        }
        res.insert(name);
    }
    return res;
}

// Gathers all referenced symbols within a statement.
// Symbols can be additionally filtered by the 'filter' predicate
template <typename Pred>
class GatherStmtSymbols : public Inspector
{
    const P4::ReferenceMap* refMap;

    // Declarations of all referenced symbols
    ordered_set<const IR::IDeclaration*> symbols;

    // Filter function.
    // Declaration is added into the result only if `pred(node) == true`
    std::optional<Pred> pred;

 public:
    GatherStmtSymbols(const P4::ReferenceMap *refMap_, Pred pred_) : refMap(refMap_), pred(pred_) {
        CHECK_NULL(refMap);
    }

    GatherStmtSymbols(const P4::ReferenceMap *refMap_) : refMap(refMap_) {
        CHECK_NULL(refMap);
    }

    ordered_set<const IR::IDeclaration*> get() const { return symbols; }

 private:
    bool preorder(const IR::Declaration* decl) {
        if (pred.has_value() && !(*pred)(decl)) {
            return true;
        }
        symbols.insert(decl);
        return true;
    }

    bool preorder(const IR::PathExpression* pe) {
        if (pred.has_value() && !(*pred)(pe)) {
            return true;
        }
        CHECK_NULL(pe->path);
        auto* decl = refMap->getDeclaration(pe->path, true);
        CHECK_NULL(decl);
        symbols.insert(decl);
        return true;
    }

    bool preorder(const IR::IfStatement* ifStmt) {
        visit(ifStmt->condition);
        return false;
    }

    bool preorder(const IR::SwitchStatement* switchStatement) {
        throw std::domain_error("Not implemented");
    }
};

} // p4mlir::tests

template <typename T>
bool operator == (const std::unordered_set<T>& lhs, const ordered_set<T>& rhs) {
    const std::unordered_set<T> tmp(rhs.begin(), rhs.end());
    return lhs == tmp;
}

template <typename T>
bool operator == (const ordered_set<T>& lhs,const std::unordered_set<T>& rhs) {
    return rhs == lhs;
}

#endif /* BACKENDS_MLIR_TESTS_COMMON_H_ */
