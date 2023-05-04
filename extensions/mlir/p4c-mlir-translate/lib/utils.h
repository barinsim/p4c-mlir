#ifndef P4C_UTILS_H
#define P4C_UTILS_H

#include <variant>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"

#include "ir/ir.h"
#include "ir/visitor.h"
#include "ir/dump.h"

namespace p4mlir {

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// P4->MLIR translation works on action/state granularity, which often demands to
// query the enclosing control/parser block for additional information.
// This class facilitates common API for both control/parser while being able to represent
// non-existent context as well. This is a value type and should be passed around by value
class BlockContext
{
    using ContextNode = std::variant<std::monostate, const IR::P4Control*, const IR::P4Parser*>;

    ContextNode node;

 public:
    BlockContext() = default;

    BlockContext(const IR::P4Control* control) {
        if (control) {
            node = control;
        }
    }

    BlockContext(const IR::P4Parser* parser) {
        if (parser) {
            node = parser;
        }
    }

    BlockContext(std::nullptr_t) {}

    operator bool() const {
        return !isEmpty();
    }

    // Returns true if represents non-existent context
    bool isEmpty() const {
        return std::holds_alternative<std::monostate>(node);
    }

    const IR::Node* toNode() const {
        return std::visit(
            overloaded{
                [](auto *arg) -> const IR::Node* { return arg; },
                [](std::monostate) -> const IR::Node * { return nullptr; },
            },
            node);
    }

    const IR::ParameterList* getApplyParameters() const {
        if (isEmpty()) {
            return nullptr;
        }
        return toNode()->to<IR::IApply>()->getApplyParameters();
    }

    const IR::ParameterList* getConstructorParameters() const {
        if (isEmpty()) {
            return nullptr;
        }
        return toNode()->to<IR::IContainer>()->getConstructorParameters();
    }

    ordered_set<const IR::IDeclaration *> getMemberVariables() const {
        if (isEmpty()) {
            return ordered_set<const IR::IDeclaration *>();
        }
        IR::IndexedVector<IR::Declaration> locals =
            std::visit(overloaded{
                           [](const IR::P4Control *arg) { return arg->controlLocals; },
                           [](const IR::P4Parser *arg) { return arg->parserLocals; },
                           [](std::monostate) { return IR::IndexedVector<IR::Declaration>(); },
                       },
                       node);
        ordered_set<const IR::IDeclaration *> members;
        std::for_each(locals.begin(), locals.end(), [&](const IR::IDeclaration *decl) {
            if (decl->is<IR::Declaration_Variable>()) {
                members.insert(decl);
            }
        });
        return members;
    }

    const IR::Type* getType() const {
        return std::visit(
            overloaded{
                [](auto *arg) -> const IR::Type* { return arg->type; },
                [](std::monostate) -> const IR::Type* { return nullptr; },
            },
            node);
    }

    std::string getName() const {
        return std::visit(
            overloaded{
                [](auto *arg) -> std::string { return arg->getName().toString().c_str(); },
                [](std::monostate) -> std::string { return ""; },
            },
            node);
    }
};

// Represents all P4 constructs that can be referenced by an MLIR symbol within the P4 dialect
using ReferenceableNode =
    std::variant<const IR::P4Action *, const IR::Method *, const IR::P4Control *>;

// Represents isolated parts of the fully qualified symbol
using SymbolParts = std::vector<mlir::StringAttr>;

// Stores the fully qualified symbols within a ModuleOp
class FullyQualifiedSymbols
{
    ordered_map<ReferenceableNode, mlir::SymbolRefAttr> data;

 public:
    void add(ReferenceableNode node, const SymbolParts& parts) {
        BUG_CHECK(!data.count(node), "Node is already resolved");
        BUG_CHECK(!parts.empty(), "Symbol is empty");

        std::vector<mlir::FlatSymbolRefAttr> tail;
        std::transform(parts.begin() + 1, parts.end(), std::back_inserter(tail), [](auto& part) {
            return mlir::FlatSymbolRefAttr::get(part);
        });
        auto root = parts.front();
        mlir::SymbolRefAttr symbol;
        if (tail.empty()) {
            symbol = mlir::SymbolRefAttr::get(root);
        } else {
            symbol = mlir::SymbolRefAttr::get(root, tail);
        }
        data.insert({node, symbol});
    }

    // Given referenceable P4 ast node, returns fully qualified 'SymbolRefAttr'
    mlir::SymbolRefAttr getSymbol(ReferenceableNode node) const {
        BUG_CHECK(data.count(node), "Node symbol does not exist");
        return data.at(node);
    }

    // Returns all containing symbols as strings.
    // Useful for testing
    std::vector<std::string> getAllAsStrings() const {
        // Converts 'SymbolRefAttr' to string
        auto toString = [](mlir::SymbolRefAttr sym) {
            std::string str = sym.getRootReference().str();
            auto tail = sym.getNestedReferences();
            std::for_each(tail.begin(), tail.end(), [&](mlir::FlatSymbolRefAttr flatSym) {
                str += std::string("::") + flatSym.getValue().str();
            });
            return str;
        };

        std::vector<std::string> rv;
        std::transform(data.begin(), data.end(), std::back_inserter(rv),
                       [&](auto &kv) { return toString(kv.second); });
        return rv;
    }
};

// Fills 'FullyQualifiedSymbols' container.
// Creates fully qualified names for P4 constructs that are referenceable by a symbol
// within a P4 dialect. P4 dialect assumes that all symbol references are fully qualified
class MakeFullyQualifiedSymbols : public Inspector
{
    mlir::OpBuilder& builder;

    // Output of this pass
    FullyQualifiedSymbols& symbols;

    // Mutable state of this pass during traversal
    // Gradually builds the current scope, adding symbol part if traversing down into a symbol table
    // and popping last symbol part while going up
    SymbolParts currentScope;

 public:
    MakeFullyQualifiedSymbols(mlir::OpBuilder& builder_, FullyQualifiedSymbols &symbols_)
        : builder(builder_), symbols(symbols_) {}

 private:
    bool preorder(const IR::P4Control* control) override {
        addToCurrentScope(control);
        symbols.add(control, currentScope);
        return true;
    }

    void postorder(const IR::P4Control*) override {
        currentScope.pop_back();
    }

    bool preorder(const IR::P4Action* action) override {
        addToCurrentScope(action);
        symbols.add(action, currentScope);
        return true;
    }

    void postorder(const IR::P4Action*) override {
        currentScope.pop_back();
    }

    bool preorder(const IR::Method* method) override {
        addToCurrentScope(method);
        symbols.add(method, currentScope);
        return true;
    }

    void postorder(const IR::Method*) override {
        currentScope.pop_back();
    }

    // Convenience method to add symbol part to the end of the 'currentScope'
    void addToCurrentScope(ReferenceableNode node) {
        cstring name =
            std::visit([](auto *arg) -> cstring { return arg->getName().toString(); }, node);
        auto strAttr = builder.getStringAttr(name.c_str());
        currentScope.push_back(strAttr);
    }

};

} // namespace p4mlir



#endif  // P4C_UTILS_H
