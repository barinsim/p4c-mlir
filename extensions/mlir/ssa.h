#ifndef BACKENDS_MLIR_SSA_H_
#define BACKENDS_MLIR_SSA_H_


#include <variant>
#include <unordered_set>
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


class GatherOutArgsScalars : public Inspector, P4WriteContext
{
    class Builder {
        std::unordered_set<const IR::IDeclaration*> decls;
     public:
        void add(const IR::IDeclaration* decl) { decls.insert(decl); }
        std::unordered_set<const IR::IDeclaration*> get() const { return decls; }
    };

    Builder b;

    const P4::ReferenceMap* refMap;
    const P4::TypeMap* typeMap;

 public:
    GatherOutArgsScalars(const P4::ReferenceMap *refMap_, const P4::TypeMap *typeMap_)
        : refMap(refMap_), typeMap(typeMap_) {
        CHECK_NULL(refMap_);
        CHECK_NULL(typeMap_);
    }

    std::unordered_set<const IR::IDeclaration*> get() const {
        return b.get();
    }

 private:
    bool preorder(const IR::PathExpression* pe) override {
        if (!isWrite() || !findContext<IR::Argument>()) {
            return true;
        }
        auto* type = typeMap->getType(pe);
        if (isPrimitiveType(type)) {
            CHECK_NULL(pe);
            b.add(refMap->getDeclaration(pe->path));
        }
        return true;
    }
};


struct RefInfo {
    std::variant<const IR::IDeclaration*, const IR::PathExpression*> ref;
    // decl == ref for declarations
    const IR::IDeclaration* decl;
};


class GatherSSAReferences : public Inspector, P4WriteContext
{
    class Builder {
        std::vector<RefInfo> reads;
        std::vector<RefInfo> writes;
     public:
        void addRead(const IR::PathExpression* pe, const IR::IDeclaration* decl) {
            CHECK_NULL(pe, decl);
            reads.push_back({pe, decl});
        }
        void addWrite(const IR::PathExpression* pe, const IR::IDeclaration* decl) {
            CHECK_NULL(pe, decl);
            writes.push_back({pe, decl});
        }
        void addWrite(const IR::IDeclaration* decl) {
             CHECK_NULL(decl);
             writes.push_back({decl, decl});
        }
        std::vector<RefInfo> getReads() const { return reads; };
        std::vector<RefInfo> getWrites() const { return writes; };
    };

    Builder b;

    const P4::TypeMap* typeMap;
    const P4::ReferenceMap* refMap;

    // Output of GatherOutParamScalars
    const std::unordered_set<const IR::IDeclaration*> forbidden;

public:
   GatherSSAReferences(const P4::TypeMap *typeMap_, const P4::ReferenceMap *refMap_,
                       std::unordered_set<const IR::IDeclaration *> forbidden_)
       : typeMap(typeMap_), refMap(refMap_), forbidden(std::move(forbidden_)) {}

   std::vector<RefInfo> getReads() const { return b.getReads(); }
   std::vector<RefInfo> getWrites() const { return b.getWrites(); }

private:
   bool preorder(const IR::PathExpression *pe) override {
        auto* type = typeMap->getType(pe);
        if (!isPrimitiveType(type)) {
            return true;
        }
        CHECK_NULL(pe->path);
        auto* decl = refMap->getDeclaration(pe->path);
        if (forbidden.count(decl)) {
            return true;
        }
        BUG_CHECK(!(isRead() && isWrite()), "ReadWrite context cannot be expressed as a SSA form");
        if (isRead()) {
            b.addRead(pe, decl);
        }
        if (isWrite()) {
            b.addWrite(pe, decl);
        }
        return true;
    }

    bool preorder(const IR::Declaration* decl) override {
        b.addWrite(decl);
        return true;
    }
};


} // namespace p4mlir


#endif /* BACKENDS_MLIR_SSA_H_ */