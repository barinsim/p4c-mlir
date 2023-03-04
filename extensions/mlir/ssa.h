#ifndef BACKENDS_MLIR_SSA_H_
#define BACKENDS_MLIR_SSA_H_


#include <stack>
#include "ir/ir.h"
#include "ir/visitor.h"
#include "frontends/common/resolveReferences/referenceMap.h"
#include "frontends/p4/typeMap.h"
#include "lib/log.h"


namespace p4mlir {

namespace {
    bool isPrimitiveType(const IR::Type* type) {
        CHECK_NULL(type);
        return type->is<IR::Type::Bits>() || type->is<IR::Type::Varbits>() || type->is<IR::Type::Boolean>();
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


/*class SSAInfo
{

 public:
    static SSAInfo* create(BasicBlock* entry, DomTree* domTree, P4ReferenceMap* refMap) {

    }

 private:
    SSAInfo()

};*/


} // namespace p4mlir


#endif /* BACKENDS_MLIR_SSA_H_ */