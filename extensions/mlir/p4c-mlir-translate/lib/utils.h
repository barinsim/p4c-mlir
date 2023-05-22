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

// P4->MLIR translation works on action/state granularity, which often demands to
// query the enclosing control/parser block for additional information.
// This class facilitates common API for both control/parser while being able to represent
// non-existent context as well. This is a value type and should be passed around by value
class BlockContext {
    using ContextNode = std::variant<std::monostate, const IR::P4Control *, const IR::P4Parser *>;

    ContextNode node;

 public:
    BlockContext() = default;

    BlockContext(const IR::P4Control *control) {
        if (control) {
            node = control;
        }
    }

    BlockContext(const IR::P4Parser *parser) {
        if (parser) {
            node = parser;
        }
    }

    BlockContext(const IR::INode* n) {
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

    BlockContext(std::nullptr_t) {}

    operator bool() const { return !isEmpty(); }

    // Returns true if represents non-existent context
    bool isEmpty() const { return std::holds_alternative<std::monostate>(node); }

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

    const IR::Type *getType() const {
        return std::visit(overloaded{
                              [](auto *arg) -> const IR::Type * { return arg->type; },
                              [](std::monostate) -> const IR::Type * { return nullptr; },
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
using ReferenceableNode = std::variant<const IR::P4Action *, const IR::Method *,
                                       const IR::P4Control *, const IR::Type_Extern *>;

// Represents isolated parts of the fully qualified symbol
using SymbolParts = std::vector<mlir::StringAttr>;

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

// Pass that converts all out-of-apply local declarations which can be referenced from an action
// into direct action parameters, which allows to move all out-of-apply local declarations into MLIR
// apply block:
//  int<16> x1 = 2;
//  foo(bit<10> arg1) {
//      x1 = 4;
//  }
//  apply {
//      foo(42);
//  }
//
// --->
//
//  int<16> __x1 = 2;
//  foo(inout int<16> __x1, bit<10> arg1) {
//      __x1 = 4;
//  }
//  apply {
//      foo(__x1, 42);
//  }
//
//  The "__" must be appended to avoid shadowing conflicts
class AddRealActionParams : public PassManager
{
    class Rename : public Transform {
        const P4::ReferenceMap *refMap = nullptr;
        ordered_map<const IR::IDeclaration *, std::string> renamed;

     public:
        Rename(const P4::ReferenceMap *refMap_) : refMap(refMap_) { CHECK_NULL(refMap); }

     private:
        IR::P4Control *preorder(IR::P4Control *control) override {
            // Rename all out-of-apply local variables to "__" + 'old name'
            IR::IndexedVector<IR::Declaration> newLocals;
            auto &locals = control->controlLocals;
            std::transform(locals.begin(), locals.end(), std::back_inserter(newLocals),
                           [&](const IR::Declaration *decl) -> const IR::Declaration * {
                               if (!decl->is<IR::Declaration_Variable>()) {
                                   return decl;
                               }
                               auto *newDecl = decl->clone();
                               std::string oldName = decl->getName().toString().c_str();
                               std::string newName = std::string("__") + oldName;
                               newDecl->name = IR::ID(newName);
                               BUG_CHECK(!renamed.count(decl), "Declaration renamed twice");
                               renamed.insert({decl, newName});
                               return newDecl;
                           });
            auto *newControl = control->clone();
            newControl->controlLocals = newLocals;
            return newControl;
        }

        IR::PathExpression *preorder(IR::PathExpression *pe) override {
            CHECK_NULL(pe->path);
            auto *decl = refMap->getDeclaration(pe->path, true);
            if (!renamed.count(decl)) {
                return pe;
            }
            IR::ID newID = pe->path->name;
            newID.name = renamed.at(decl);
            return new IR::PathExpression(pe->srcInfo, pe->type, new IR::Path(newID));
        }

        IR::P4Parser *preorder(IR::P4Parser *parser) override { BUG_CHECK(false, "Not implemented"); }
    };

    class CollectAdditionalActionParams : public Inspector {
     public:
        using ContainerType =
            ordered_map<const IR::P4Action *, std::vector<const IR::Declaration_Variable *>>;

        CollectAdditionalActionParams(ContainerType &additionalParams_)
            : additionalParams(additionalParams_) {}

     private:
        ContainerType &additionalParams;

        bool preorder(const IR::P4Action *action) override {
            additionalParams.insert({action, {}});
            auto *context = findContext<IR::P4Control>();
            if (!context) {
                return true;
            }
            // Add all out-of-apply variable declarations that are above this action
            auto &locals = context->controlLocals;
            for (auto it = locals.begin(); it != locals.end(); ++it) {
                if (*it == action) {
                    break;
                }
                if (auto *decl = (*it)->to<IR::Declaration_Variable>()) {
                    additionalParams.at(action).push_back(decl);
                }
            }

            return true;
        }
    };

    class PrependAdditionalActionParams : public Transform {
        const CollectAdditionalActionParams::ContainerType &additionalParams;

     public:
        PrependAdditionalActionParams(
            const CollectAdditionalActionParams::ContainerType &additionalParams_)
            : additionalParams(additionalParams_) {}

     private:
        IR::ParameterList *preorder(IR::ParameterList *params) override {
            auto *action = findOrigCtxt<IR::P4Action>();
            if (!action) {
                return params;
            }

            // Prepend additional parameters to the parameter list
            // TODO: the InOut direction is overly conservative and should depend on the usage of the
            // parameter within the action
            auto *newParams = new IR::ParameterList();
            auto &toPrepend = additionalParams.at(action);
            std::for_each(
                toPrepend.begin(), toPrepend.end(), [&](const IR::Declaration_Variable *decl) {
                    cstring name = decl->getName();
                    auto *newParam = new IR::Parameter(name, IR::Direction::InOut, decl->type);
                    newParams->push_back(newParam);
                });
            std::for_each(params->begin(), params->end(),
                          [&](auto *param) { newParams->push_back(param); });

            return newParams;
        }
    };

    class FixActionCallSites : public Transform {
        const CollectAdditionalActionParams::ContainerType &additionalParams;
        P4::ReferenceMap *refMap = nullptr;
        P4::TypeMap *typeMap = nullptr;

     public:
        FixActionCallSites(const CollectAdditionalActionParams::ContainerType &additionalParams_,
                           P4::ReferenceMap *refMap_, P4::TypeMap *typeMap_)
            : additionalParams(additionalParams_), refMap(refMap_), typeMap(typeMap_) {
            CHECK_NULL(refMap, typeMap);
        }

        IR::MethodCallExpression *preorder(IR::MethodCallExpression *call) override {
            auto *instance = P4::MethodInstance::resolve(call, refMap, typeMap);
            auto *actCall = instance->to<P4::ActionCall>();
            if (!actCall) {
                return call;
            }

            // Create new arguments by prepending additional ones in front of the old ones
            auto &toPrepend = additionalParams.at(actCall->action);
            auto *oldArgs = call->arguments;
            auto *newArgs = new IR::Vector<IR::Argument>();
            std::for_each(toPrepend.begin(), toPrepend.end(),
                          [&](const IR::Declaration_Variable *decl) {
                              // TODO: the name can be shadowed here, rename
                              cstring name = decl->getName();
                              auto *newPath = new IR::PathExpression(name);
                              auto *newArg = new IR::Argument(newPath);
                              newArgs->push_back(newArg);
                          });
            std::copy(oldArgs->begin(), oldArgs->end(), std::back_inserter(*newArgs));

            return new IR::MethodCallExpression(call->method, call->typeArguments, newArgs);
        }
    };

 public:
    AddRealActionParams() {
        auto* additionalParams = new CollectAdditionalActionParams::ContainerType();
        auto* refMap = new P4::ReferenceMap();
        auto* typeMap = new P4::TypeMap();

        addPasses({
            new P4::ResolveReferences(refMap),
            new Rename(refMap),
            new P4::ResolveReferences(refMap),
            new P4::TypeInference(refMap, typeMap, false, true),
            new P4::TypeChecking(refMap, typeMap, true),
            new CollectAdditionalActionParams(*additionalParams),
            new PrependAdditionalActionParams(*additionalParams),
            new FixActionCallSites(*additionalParams, refMap, typeMap),
            new P4::ClearTypeMap(typeMap, true),
            new P4::ResolveReferences(refMap),
            new P4::TypeInference(refMap, typeMap, false, true),
            new P4::TypeChecking(refMap, typeMap, true)
        });
    }
};

} // namespace p4mlir



#endif  // P4C_UTILS_H
