#include "gtest/gtest.h"

#include <unordered_set>
#include "test/gtest/helpers.h"
#include "frontends/common/parseInput.h"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/toP4/toP4.h"
#include "common.h"
#include "../cfgBuilder.h"
#include "../domTree.h"
#include "../ssa.h"


namespace p4mlir::tests {


class SSAInfo : public Test::P4CTest { };


TEST_F(SSAInfo, Gather_out_param_scalars_collects_inout_and_out_args) {
    std::string src = P4_SOURCE(R"(
        extern int<16> bar1(inout int<16> x1, in int<16> x2);
        extern int<16> bar2(out int<16> x1);
        action foo() {
            int<16> f1 = 3;
            int<16> f2 = 3;
            int<16> f3 = 3;
            int<16> f4 = 3;
            bar1(f1, bar1(f2, f3));
            bar2(f4);
        }
    )");
    auto* program = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(program && ::errorCount() == 0);

    auto* refMap = new P4::ReferenceMap();
    auto* typeMap = new P4::TypeMap();
    program = program->apply(P4::ResolveReferences(refMap));
    program = program->apply(P4::TypeInference(refMap, typeMap, false, true));
    program = program->apply(P4::TypeChecking(refMap, typeMap, true));
    ASSERT_TRUE(program && ::errorCount() == 0);

    auto* g = new p4mlir::GatherOutParamScalars(refMap, typeMap);
    program->apply(*g);
    ASSERT_TRUE(program && ::errorCount() == 0);

    auto names = [](std::unordered_set<const IR::IDeclaration*> decls) {
        std::unordered_set<cstring> res;
        for (auto* d : decls) {
            res.insert(d->getName().name);
        }
        return res;
    };

    using unordered = std::unordered_set<cstring>;
    EXPECT_EQ(names(g->get()), unordered({"f1", "f2", "f4"}));
}


} // namespace p4mlir::tests
