#ifndef P4C_UTILS_H
#define P4C_UTILS_H

#include <variant>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"

#include "frontends/p4/methodInstance.h"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/typeChecking/typeChecker.h"

#include "ir/ir.h"
#include "ir/visitor.h"
#include "ir/dump.h"
#include "ir/pass_manager.h"

namespace p4mlir {

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// Common API for P4Control and P4Parser. Can represent missing block as well.
// Throughout the P4->MLIR translation process some algorithms can be reused for both P4Control and
// P4Parser. This class facilitates common API that is not provided by the P4 AST. This class is a
// value type and should be passed around by value
class P4Block {
    using BlockNode = std::variant<std::monostate, const IR::P4Control *, const IR::P4Parser *>;

    BlockNode node;

 public:
    P4Block() = default;

    P4Block(const IR::INode* n) {
        if (n) {
            if (auto* control = n->to<IR::P4Control>()) {
                node = control;
            } else if (auto* parser = n->to<IR::P4Parser>()) {
                node = parser;
            } else {
              BUG_CHECK(false, "Expected P4Control or P4Parser");
            }
        }
    }

    operator bool() const { return !isEmpty(); }

    // Returns true if represents non-existent block
    bool isEmpty() const { return std::holds_alternative<std::monostate>(node); }

    bool isControl() const { return std::holds_alternative<const IR::P4Control*>(node); }
    bool isParser() const { return std::holds_alternative<const IR::P4Parser*>(node); }

    const IR::Node *toNode() const {
        return std::visit(overloaded{
            [](auto *arg) -> const IR::Node * { return arg; },
            [](std::monostate) -> const IR::Node * { return nullptr; },
        },
        node);
    }

    const IR::ParameterList *getApplyParameters() const {
        if (isEmpty()) {
            return nullptr;
        }
        return toNode()->to<IR::IApply>()->getApplyParameters();
    }

    const IR::ParameterList *getConstructorParameters() const {
        if (isEmpty()) {
            return nullptr;
        }
        return toNode()->to<IR::IContainer>()->getConstructorParameters();
    }

    const IR::Type_MethodBase* getConstructorType() const {
        if (isEmpty()) {
            return nullptr;
        }
        return toNode()->to<IR::IContainer>()->getConstructorMethodType();
    }

    const IR::Type_MethodBase* getApplyMethodType() const {
        if (isEmpty()) {
            return nullptr;
        }
        return toNode()->to<IR::IApply>()->getApplyMethodType();
    }

    std::vector<const IR::IDeclaration *> getBodyDeclarations() const {
        std::vector<const IR::IDeclaration*> decls = std::visit(overloaded{
            [](const IR::P4Control *control) {
               std::vector<const IR::IDeclaration *> rv(control->controlLocals.begin(),
                                                        control->controlLocals.end());
               return rv;
            },
            [](const IR::P4Parser *parser) {
               // For parser we return parserLocals + states
               std::vector<const IR::IDeclaration *> rv(parser->parserLocals.begin(),
                                                        parser->parserLocals.end());
               rv.insert(rv.end(), parser->states.begin(), parser->states.end());
               return rv;
            },
            [](std::monostate) { return std::vector<const IR::IDeclaration*>(); },
        },
        node);
        return decls;
    }

    const IR::Type *getType() const {
        return std::visit(overloaded{
            [](auto *arg) -> const IR::Type * { return arg->type; },
            [](std::monostate) -> const IR::Type * { return nullptr; },
        },
        node);
    }

    std::string getName() const {
        return std::visit(overloaded{
            [](auto *arg) -> std::string { return arg->getName().toString().c_str(); },
            [](std::monostate) -> std::string { return ""; },
        },
        node);
    }
};

// Represents all P4 constructs that can be referenced by an MLIR symbol within the P4 dialect
using ReferenceableNode =
    std::variant<const IR::P4Action *, const IR::Method *, const IR::P4Control *,
                 const IR::Type_Extern *, const IR::Declaration_ID *>;

// Represents isolated parts of the fully qualified symbol
using SymbolParts = std::vector<mlir::StringAttr>;

// Container for the MakeFullyQualifiedSymbols pass.
// Stores the fully qualified symbols within a ModuleOp
class FullyQualifiedSymbols
{
    ordered_map<ReferenceableNode, mlir::SymbolRefAttr> data;

 public:
    void add(ReferenceableNode node, const SymbolParts &parts) {
        BUG_CHECK(!parts.empty(), "Symbol is empty");

        // Create symbol out of strings
        std::vector<mlir::FlatSymbolRefAttr> tail;
        std::transform(parts.begin() + 1, parts.end(), std::back_inserter(tail),
                       [](auto &part) { return mlir::FlatSymbolRefAttr::get(part); });
        auto root = parts.front();
        mlir::SymbolRefAttr symbol;
        if (tail.empty()) {
            symbol = mlir::SymbolRefAttr::get(root);
        } else {
            symbol = mlir::SymbolRefAttr::get(root, tail);
        }

        if (data.count(node)) {
            BUG_CHECK(data.at(node) == symbol,
                      "Node already exists and the added symbol is not duplicate");
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
    mlir::OpBuilder &builder;

    const P4::TypeMap* typeMap = nullptr;

    // Output of this pass
    FullyQualifiedSymbols &symbols;

    // Mutable state of this pass during traversal
    // Gradually builds the current scope, adding symbol part if traversing down into a symbol table
    // and popping last symbol part while going up
    SymbolParts currentScope;

 public:
    MakeFullyQualifiedSymbols(mlir::OpBuilder &builder_, FullyQualifiedSymbols &symbols_,
                              const P4::TypeMap *typeMap_)
        : builder(builder_), symbols(symbols_), typeMap(typeMap_) {
        CHECK_NULL(typeMap_);
    }

 private:
    bool preorder(const IR::P4Control *control) override {
        addToCurrentScope(control);
        symbols.add(control, currentScope);
        return true;
    }

    void postorder(const IR::P4Control *) override { currentScope.pop_back(); }

    bool preorder(const IR::P4Action *action) override {
        addToCurrentScope(action);
        symbols.add(action, currentScope);
        return true;
    }

    void postorder(const IR::P4Action *) override { currentScope.pop_back(); }

    bool preorder(const IR::Method *method) override {
        // To solve P4 overloading of methods, the names of methods in MLIR consist of the original
        // P4 name + '_' + <number of parameters>
        std::size_t numParams = method->getParameters()->size();
        std::string newName = method->getName().toString() + "_" + std::to_string(numParams);
        auto strAttr = builder.getStringAttr(newName);
        currentScope.push_back(strAttr);
        symbols.add(method, currentScope);
        return true;
    }

    void postorder(const IR::Method *) override { currentScope.pop_back(); }

    bool preorder(const IR::Type_Extern *ext) override {
        addToCurrentScope(ext);
        symbols.add(ext, currentScope);
        return true;
    }

    void postorder(const IR::Type_Extern *) override { currentScope.pop_back(); }

    bool preorder(const IR::Declaration_ID* declID) override {
        addToCurrentScope(declID);
        symbols.add(declID, currentScope);
        return true;
    }

    void postorder(const IR::Declaration_ID *) override { currentScope.pop_back(); }

    bool preorder(const IR::Declaration_Instance* decl) override {
        // Instantiated externs are not visited by default.
        // Visit them when we encounter their instantiations
        auto* type = typeMap->getType(decl, true);
        auto saved = currentScope;
        currentScope.clear();
        visit(type);
        currentScope = saved;
        return true;
    }

    // Convenience method to add symbol part to the end of the 'currentScope'
    void addToCurrentScope(ReferenceableNode node) {
        cstring name =
            std::visit([](auto *arg) -> cstring { return arg->getName().toString(); }, node);
        auto strAttr = builder.getStringAttr(name.c_str());
        currentScope.push_back(strAttr);
    }
};

// Container for the 'CollectAdditionalParams' pass.
// Maps table/action to the declarations of its additional parameters
class AdditionalParams
{
    // Represents types which might need additional parameters
    using Callable = std::variant<const IR::P4Action *, const IR::P4Table *>;

    ordered_map<Callable, std::vector<const IR::Declaration_Variable *>> data;

 public:
    void add(Callable callable, std::vector<const IR::Declaration_Variable*> decls) {
        BUG_CHECK(!data.count(callable), "Params for a callable added twice");
        data[callable] = decls;
    }

    std::vector<const IR::Declaration_Variable *> get(Callable callable) const {
        BUG_CHECK(data.count(callable), "Unknown callable");
        return data.at(callable);
    }
};

// Translation into P4 dialect makes some implicit parameters explicit. This pass collects these
// extra parameters which are then considered during MLIRgen. Currently collected extra parameters
// are out-of-apply local variables which can be referenced from within a table/action definition.
// Currently, it is assumed that all out-of-apply local variables are allocated onto a stack,
// which heavily simplifies things since no SSA numbering has to be computed for them
class CollectAdditionalParams : public Inspector
{
    // Output of this pass
    AdditionalParams& data;

    // Running list of local out-of-apply variables
    std::vector<const IR::Declaration_Variable*> locals;

    bool preorder(const IR::P4Control* control) override {
        auto& controlLocals = control->controlLocals;
        std::for_each(controlLocals.begin(), controlLocals.end(), [&](const IR::Declaration *decl) {
            if (auto* local = decl->to<IR::Declaration_Variable>()) {
                locals.push_back(local);
            } else {
                visit(decl);
            }
        });
        return false;
    }

    void postorder(const IR::P4Control* control) override {
        // Clear the list of locals while leaving the control
        locals.clear();
    }

    bool preorder(const IR::P4Parser* parser) override {
        auto& parserLocals = parser->parserLocals;
        std::for_each(parserLocals.begin(), parserLocals.end(), [&](const IR::Declaration *decl) {
            if (auto* local = decl->to<IR::Declaration_Variable>()) {
                locals.push_back(local);
            } else {
                visit(decl);
            }
        });
        return false;
    }

    void postorder(const IR::P4Parser* parser) override {
        // Clear the list of locals while leaving the parser
        locals.clear();
    }

    bool preorder(const IR::P4Action* action) override {
        data.add(action, locals);
        return false;
    }

    bool preorder(const IR::P4Table* table) override {
        data.add(table, locals);
        return false;
    }

public:
   CollectAdditionalParams(AdditionalParams& data_) : data(data_) {}
};

} // namespace p4mlir



#endif  // P4C_UTILS_H
