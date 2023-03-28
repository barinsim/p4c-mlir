#ifndef BACKENDS_MLIR_TESTS_COMMON_H_
#define BACKENDS_MLIR_TESTS_COMMON_H_


#include <string>

#include "lib/ordered_set.h"
#include "lib/ordered_map.h"

#include "cfgBuilder.h"


namespace p4mlir::tests {


BasicBlock* getByName(const std::map<const IR::IDeclaration*, BasicBlock*>&, const std::string&);

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
