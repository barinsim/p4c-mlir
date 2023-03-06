#ifndef BACKENDS_MLIR_SSA_H_
#define BACKENDS_MLIR_SSA_H_


#include <stack>
#include "ir/ir.h"
#include "ir/visitor.h"
#include "ir/dump.h"
#include "frontends/common/resolveReferences/referenceMap.h"
#include "frontends/p4/typeMap.h"
#include "lib/log.h"


namespace p4mlir {


namespace {

bool isPrimitiveType(const IR::Type *type) {
    CHECK_NULL(type);
    return type->is<IR::Type::Bits>() || type->is<IR::Type::Varbits>() ||
           type->is<IR::Type::Boolean>();
}

}


class Declaration
{
    const IR::IDeclaration* instance;
    std::optional<std::vector<const IR::StructField*>> field;

 public:
    static Declaration create(const IR::PathExpression *pe, const P4::TypeMap *typeMap,
                  const P4::ReferenceMap *refMap);
    static Declaration create(const IR::Member *mem, const P4::TypeMap *typeMap,
                  const P4::ReferenceMap *refMap);

    std::string toString() const;

private:
    Declaration() = delete;
    Declaration(const IR::IDeclaration *instance_) : instance(instance_) { CHECK_NULL(instance); }
    Declaration(const IR::IDeclaration *instance_, std::vector<const IR::StructField *> field_)
        : instance(instance_), field(std::move(field_)) {
        CHECK_NULL(instance);
        std::for_each(field->begin(), field->end(), [](auto* f) { CHECK_NULL(f); });
    }
};


class GatherSSAReferences : public Inspector
{
    class Builder {
        std::vector<Declaration> refs;
     public:
        void add(Declaration decl) { refs.push_back(decl); }
        std::vector<Declaration> get() const { return refs; };
    };

    Builder b;

    const P4::TypeMap* typeMap;
    const P4::ReferenceMap* refMap;

 public:
    GatherSSAReferences(const P4::TypeMap *typeMap_, const P4::ReferenceMap *refMap_)
        : typeMap(typeMap_), refMap(refMap_) {}

    std::vector<Declaration> getSSARefs() const { return b.get(); }

 private:
    bool preorder(const IR::PathExpression* pe) override {
        auto* type = typeMap->getType(pe);
        if (type && isPrimitiveType(type)) {
            LOG3("GatherRefs: adding " << pe);
            b.add(Declaration::create(pe, typeMap, refMap));
        }
        return true;
    }

    bool preorder(const IR::Member* mem) override {
        // Field references of type
        // <StructLike>.<StructLike>.<StructLike>.<PrimitiveType>
        // can be converted into SSA values, since they cannot be aliased.
        // Contrary to references like hdr.hs[idx].f1 or table.apply().hit.
        auto cannotBeAliased = [&](const IR::Member* m) {
            CHECK_NULL(m->expr);
            const IR::Expression* ptr = m->expr;
            while (ptr->is<IR::Member>()) {
                auto* base = ptr->to<IR::Member>()->expr;
                CHECK_NULL(base);
                auto* type = typeMap->getType(base);
                if (!type || !type->is<IR::Type_StructLike>()) {
                    return false;
                }
                ptr = base;
            }
            auto* type = typeMap->getType(ptr);
            return ptr->is<IR::PathExpression>() && type && type->is<IR::Type_StructLike>();
        };

        auto* type = typeMap->getType(mem);
        if (type && isPrimitiveType(type) && cannotBeAliased(mem)) {
            LOG3("GatherRefs: adding " << mem);
            b.add(Declaration::create(mem, typeMap, refMap));
        }
        return true;
    }
};


class SSAConversion : public Transform
{
    // New statements/declarations that were created during visiting current statement/declaration.
    // Gets copied into 'toInsert' and cleared after each statement.
    std::vector<IR::StatOrDecl*> newStmts;

    // New statements for current block statement. Maps insertion point to statements that need to
    // be inserted before the insertion point.
    std::unordered_map<const IR::StatOrDecl*, std::vector<IR::StatOrDecl*>> toInsert;

    const P4::TypeMap* typeMap = nullptr;

 public:
    SSAConversion(const P4::TypeMap* typeMap_) : typeMap(typeMap_) { CHECK_NULL(typemap); }

 private:
    IR::Node* preorder(IR::StatOrDecl* statOrDecl) {
        newStmts.clear();
        return statOrDecl;
    }

    IR::Node* postorder(IR::StatOrDecl* statOrDecl) {
        if (!findContext<IR::BlockStatement>()) {
            BUG_CHECK(newStmts.empty(), "Rewriting outside of block statement is not implemented");
            return statOrDecl;
        }
        BUG_CHECK(!toInsert.count(statOrDecl), "Statement visited twice");
        toInsert.insert({statOrDecl, std::move(newStmts)});
        newStmts.clear();
        return statOrDecl;
    }

    IR::Node* postorder(IR::MethodCallExpression* call) {
        auto* baseType = typeMap->getType(call->method)->to<IR::Type_MethodBase>();
        auto* retType = baseType->returnType;
        bool hasRetVal = retType && !retType->is<IR::Type_Void>();
        IR::Declaration_Variable* decl = nullptr;
        if (hasRetVal && !getParent<IR::MethodCallStatement>()) {
            // TODO: generate the name from refMap
            static int id = 0;
            std::string name = "__tmp" + std::to_string(id++);
            decl = new IR::Declaration_Variable(IR::ID(name), retType);
        }
        if (decl) {
            newStmts.push_back(decl);
            auto* ref = new IR::PathExpression(decl->getName());
            auto* refs = new IR::Vector<IR::Expression>({ref});
            newStmts.push_back(new IR::SSACall(call, refs));
            return new IR::PathExpression(decl->getName());
        }
        newStmts.push_back(new IR::SSACall(call, new IR::Vector<IR::Expression>()));
        return nullptr;
    }

    IR::Node* postorder(IR::MethodCallStatement* stmt) {
        BUG_CHECK(!newStmts.empty() && newStmts.back()->is<IR::SSACall>(),
                  "MethodCallStatement should always produce SSACall");
        auto* ssaCall = newStmts.back();
        newStmts.pop_back();
        BUG_CHECK(!toInsert.count(stmt), "Statement visited twice");
        toInsert.insert({ssaCall, std::move(newStmts)});
        newStmts.clear();
        return ssaCall;
    }

    IR::Node* preorder(IR::BlockStatement* block) {
        toInsert.clear();
        return block;
    }

    IR::Node* postorder(IR::BlockStatement* block) {
        if (toInsert.empty()) {
            return block;
        }
        auto* newBlock = new IR::BlockStatement(block->srcInfo, block->annotations);
        auto& components = newBlock->components;
        for (auto* comp : block->components) {
            if (toInsert.count(comp)) {
                components.insert(components.end(), toInsert[comp].begin(), toInsert[comp].end());
                toInsert.erase(comp);
            }
            components.push_back(comp);
        }
        BUG_CHECK(toInsert.empty(), "Some of the new statements were not inserted");
        return newBlock;
    }
};


} // namespace p4mlir


#endif /* BACKENDS_MLIR_SSA_H_ */