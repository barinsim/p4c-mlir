#include "ssa.h"


namespace p4mlir {


Declaration Declaration::create(const IR::PathExpression *pe, const P4::TypeMap *typeMap,
                                const P4::ReferenceMap *refMap) {
    CHECK_NULL(pe, typeMap, refMap);
    return Declaration(refMap->getDeclaration(pe->path));
}

Declaration Declaration::create(const IR::Member *mem, const P4::TypeMap *typeMap,
                  const P4::ReferenceMap *refMap) {
    CHECK_NULL(mem, typeMap, refMap);
    std::stack<const IR::Member*> comps;
    const IR::Expression* root = mem;
    while (root->is<IR::Member>()) {
        auto* m = root->to<IR::Member>();
        comps.push(m);
        root = m->expr;
    }
    BUG_CHECK(root->is<IR::PathExpression>(), "%1% must be a path expression", root);
    BUG_CHECK(isPrimitiveType(typeMap->getType(comps.top())),
              "Last member of a path must be refer a primitive type");
    std::vector<const IR::StructField*> fields;
    while (!comps.empty()) {
        auto* curr = comps.top();
        comps.pop();
        CHECK_NULL(curr, curr->expr);
        auto* type = typeMap->getType(curr->expr);
        CHECK_NULL(type);
        auto* baseType = type->to<IR::Type_StructLike>();
        CHECK_NULL(baseType);
        auto* field = baseType->getField(curr->member);
        CHECK_NULL(field);
        fields.push_back(field);
    }
    return Declaration(refMap->getDeclaration(root->to<IR::PathExpression>()->path), fields);
}

std::string Declaration::toString() const {
    CHECK_NULL(instance);
    std::stringstream ss;
    ss << instance->getName();
    if (field.has_value()) {
        for (auto* f : field.value()) {
            ss << "." << f->getName();
        }
    }
    return ss.str();
}

} // namespace p4mlir