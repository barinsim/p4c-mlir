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


TEST_F(SSAInfo, Correctly_detect_SSA_reads_and_writes_considering_out_args) {
    std::string src = P4_SOURCE(R"(
        extern int<16> bar1(inout int<16> x1, in int<16> x2);
        extern int<16> bar2(out int<16> x1);
        extern int<16> bar3(in int<16> x1, in int<16> x2, in int<16> x3);
        action foo() {
            int<16> f1 = 3;
            int<16> f2 = 3;
            int<16> f3 = 3;
            int<16> f4 = 3;
            int<16> f5 = 3;
            int<16> f6 = f5;
            bar1(f1, bar1(f2, f3));
            bar2(f4);
            bar3(f1, f5, f6);
            f5 = f3 + f1;
            f6 = f3 + f6;
            return;
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

    auto* cfgBuilder = new p4mlir::CFGBuilder();
    program->apply(*cfgBuilder);
    auto cfg = cfgBuilder->getCFG();
    ASSERT_EQ(cfg.size(), (std::size_t)1);
    auto* foo = cfg.begin()->second;
    auto* fooAST = cfg.begin()->first->to<IR::P4Action>();
    ASSERT_TRUE(foo && program && ::errorCount() == 0);

    auto* g = new p4mlir::GatherOutArgsScalars(refMap, typeMap);
    fooAST->apply(*g);
    ASSERT_TRUE(program && ::errorCount() == 0);

    auto names = [](auto decls) {
        std::unordered_set<cstring> res;
        for (auto* d : decls) {
            res.insert(d->getName().name);
        }
        return res;
    };

    using unordered = std::unordered_set<cstring>;
    EXPECT_EQ(names(g->get()), unordered({"f1", "f2", "f4"}));

    auto forbidden = g->get();

    auto writes = [&](auto* stmt) {
        p4mlir::GatherSSAReferences refs(typeMap, refMap, forbidden);
        stmt->apply(refs);
        std::vector<const IR::IDeclaration*> decls;
        auto w = refs.getWrites();
        std::transform(w.begin(), w.end(), std::back_inserter(decls),
                       [](auto& x) { return x.decl; });
        return names(decls);
    };

    auto reads = [&](auto* stmt) {
        p4mlir::GatherSSAReferences refs(typeMap, refMap, forbidden);
        stmt->apply(refs);
        std::vector<const IR::IDeclaration*> decls;
        auto r = refs.getReads();
        std::transform(r.begin(), r.end(), std::back_inserter(decls),
                       [](auto& x) { return x.decl; });
        return names(decls);
    };

    auto stmtIt = foo->components.begin();

    // int<16> f1 = 3;
    EXPECT_EQ(writes(*stmtIt), unordered({"f1"}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // int<16> f2 = 3;
    EXPECT_EQ(writes(*stmtIt), unordered({"f2"}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // int<16> f3 = 3;
    EXPECT_EQ(writes(*stmtIt), unordered({"f3"}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // int<16> f4 = 3;
    EXPECT_EQ(writes(*stmtIt), unordered({"f4"}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // int<16> f5 = 3;
    EXPECT_EQ(writes(*stmtIt), unordered({"f5"}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    //int<16> f6 = f5;
    EXPECT_EQ(writes(*stmtIt), unordered({"f6"}));
    EXPECT_EQ(reads(*stmtIt), unordered({"f5"}));
    stmtIt++;
    // bar1(f1, bar1(f2, f3));
    EXPECT_EQ(writes(*stmtIt), unordered({}));
    EXPECT_EQ(reads(*stmtIt), unordered({"f3"}));
    stmtIt++;
    // bar2(f4);
    EXPECT_EQ(writes(*stmtIt), unordered({}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // bar3(f1, f5, f6)
    EXPECT_EQ(writes(*stmtIt), unordered({}));
    EXPECT_EQ(reads(*stmtIt), unordered({"f5", "f6"}));
    stmtIt++;
    // f5 = f3 + f1;
    EXPECT_EQ(writes(*stmtIt), unordered({"f5"}));
    EXPECT_EQ(reads(*stmtIt), unordered({"f3"}));
    stmtIt++;
    // f6 = f3 + f6;
    EXPECT_EQ(writes(*stmtIt), unordered({"f6"}));
    EXPECT_EQ(reads(*stmtIt), unordered({"f3", "f6"}));
    stmtIt++;
    // return;
    EXPECT_EQ(writes(*stmtIt), unordered({}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    EXPECT_EQ(stmtIt, foo->components.end());

    ASSERT_TRUE(program && ::errorCount() == 0);
}


} // namespace p4mlir::tests
