#include "common.h"

#include <exception>
#include <string>
#include <algorithm>


namespace p4mlir::tests {


BasicBlock* getByName(const std::map<const IR::Node*, BasicBlock*>& cfg,
                      const std::string& name) {
    auto it = std::find_if(cfg.begin(), cfg.end(), [&](auto& p) {
        auto* node = p.first;
        if (auto* decl = node->template to<IR::IDeclaration>()) {
            return decl->getName() == name;
        }
        if (auto* bs = node->template to<IR::BlockStatement>()) {
            // Apply method can be found using empty string
            return name == "";
        }
    });
    if (it == cfg.end()) {
        throw std::domain_error("The declaration name does not exist in the cfg");
    }
    return it->second;
}

p4mlir::BasicBlock* getByStmtString(p4mlir::BasicBlock* entry, const std::string& stmt) {
    auto bbs = CFGWalker::collect(entry, [&](auto* bb) {
        return std::any_of(bb->components.begin(), bb->components.end(), [&](auto* c) {
            return CFGPrinter::toString(c) == stmt;
        });
    });
    if (bbs.size() != 1) {
        throw std::domain_error("The searched statement must be unique and must exist");
    }
    return bbs.front();
}

} // p4mlir::tests
