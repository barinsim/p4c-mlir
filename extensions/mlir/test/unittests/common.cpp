#include "common.h"

#include <exception>
#include <string>
#include <algorithm>


namespace p4mlir::tests {


p4mlir::CFG getByName(const p4mlir::CFGInfo& cfgInfo,
                      const std::string& name) {
    auto it = std::find_if(cfgInfo.begin(), cfgInfo.end(), [&](auto& p) {
        auto* node = p.first;
        if (auto* decl = node->template to<IR::IDeclaration>()) {
            return decl->getName() == name;
        }
        return false;
    });
    if (it == cfgInfo.end()) {
        throw std::domain_error("The declaration name does not exist in the cfg");
    }
    return it->second;
}

p4mlir::BasicBlock* getByStmtString(CFG cfg, const std::string& stmt) {
    auto bbs = CFGWalker::collect(cfg.getEntry(), [&](auto* bb) {
        return std::any_of(bb->components.begin(), bb->components.end(), [&](auto* c) {
            return CFGPrinter::toString(c) == stmt;
        });
    });
    if (bbs.size() != 1) {
        throw std::domain_error("The searched statement must be unique and must exist");
    }
    return bbs.front();
}

std::string tokenize(const std::string& str) {
    if (str.empty()) {
        return str;
    }
    std::string rv = str;
    for (auto separator : {"\n", " "}) {
        boost::char_separator<char> sep(separator);
        boost::tokenizer<boost::char_separator<char>> tok(rv, sep);
        std::string res = *tok.begin();
        std::for_each(++tok.begin(), tok.end(), [&](auto& t) {
            res.append(std::string(" ") + t);
        });
        rv = res;
    }
    return rv;
}

ParseOutput parseP4ForTests(const std::string& p4string) {
    auto* program = P4::parseP4String(p4string, CompilerOptions::FrontendVersion::P4_16);
    if (::errorCount() > 0) {
        return ParseOutput{.ast = program};
    }
    auto* refMap = new P4::ReferenceMap();
    auto* typeMap = new P4::TypeMap();
    program = program->apply(P4::ResolveReferences(refMap));
    program = program->apply(P4::TypeInference(refMap, typeMap, false, true));
    program = program->apply(P4::TypeChecking(refMap, typeMap, true));
    return ParseOutput{.ast = program, .typeMap = typeMap, .refMap = refMap};
}

TestMLIRContext createMLIRContext() {
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<p4mlir::P4Dialect>();
    auto builder = std::make_unique<mlir::OpBuilder>(context.get());
    return TestMLIRContext{.builder = std::move(builder), .context = std::move(context)};
}

} // p4mlir::tests
