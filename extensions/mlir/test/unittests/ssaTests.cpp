#include "gtest/gtest.h"

#include <unordered_set>
#include "test/gtest/helpers.h"
#include "frontends/common/parseInput.h"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/toP4/toP4.h"
#include "common.h"
#include "cfgBuilder.h"
#include "domTree.h"
#include "ssa.h"


namespace p4mlir::tests {


class SSAInfo : public Test::P4CTest { };


TEST_F(SSAInfo, Test_ssa_conversion_for_simple_action) {
    std::string src = P4_SOURCE(R"(
        action foo() {
            int<16> x = 3;
            int<16> res;
            if (x > 3) {
                x = 2;
            } else {
                x = 1;
            }
            res = x;
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
    ASSERT_TRUE(::errorCount() == 0);

    p4mlir::SSAInfo ssaInfo(*cfg.begin(), refMap, typeMap);
    ASSERT_TRUE(::errorCount() == 0);

    auto* foo = getByName(cfg, "foo");

    auto names = [](auto decls) {
        std::unordered_set<cstring> res;
        for (auto[d, p] : decls) {
            res.insert(d->getName().name);
        }
        return res;
    };

    auto* bb1 = getByStmtString(foo, "int<16> x = (int<16>)16s3;");
    auto* bb2 = getByStmtString(foo, "x = (int<16>)16s2;");
    auto* bb3 = getByStmtString(foo, "x = (int<16>)16s1;");
    auto* bb4 = getByStmtString(foo, "res = x;");

    using unordered = std::unordered_set<cstring>;
    EXPECT_EQ(names(ssaInfo.getPhiInfo(bb1)), unordered({}));
    EXPECT_EQ(names(ssaInfo.getPhiInfo(bb2)), unordered({}));
    EXPECT_EQ(names(ssaInfo.getPhiInfo(bb3)), unordered({}));
    EXPECT_EQ(names(ssaInfo.getPhiInfo(bb4)), unordered({"x"}));

    // action foo() {
    //     int<16> x$0 = 3;
    //     int<16> res$0;
    //     if (x$0 > 3) {
    //         x$1 = 2;
    //     } else {
    //         x$2 = 1;
    //     }
    //     x$3 = phi(x$1, x$2)
    //     res$1 = x$3;
    //     return;
    // }

    auto phiInfo = ssaInfo.getPhiInfo(bb4);
    p4mlir::SSAInfo::Phi phi = phiInfo.begin()->second;

    EXPECT_TRUE(phi.destination.has_value());
    EXPECT_TRUE(phi.sources.at(bb2).has_value());
    EXPECT_TRUE(phi.sources.at(bb3).has_value());

    // This relies on a single def/use within a statement
    auto writeOrReadID = [&](const IR::StatOrDecl* stmt, bool reads) {
        p4mlir::GatherSSAReferences refs(typeMap, refMap, {});
        stmt->apply(refs);
        std::vector<p4mlir::RefInfo> infos;
        if (reads) {
            infos = refs.getReads();
        } else {
            infos = refs.getWrites();
        }
        EXPECT_EQ(infos.size(), (std::size_t)1);
        auto info = infos.front();
        EXPECT_TRUE(ssaInfo.isSSARef(info.ref));
        return ssaInfo.getID(info.ref);
    };
    auto writeID = [&](const IR::StatOrDecl* stmt) { return writeOrReadID(stmt, false); };
    auto readID = [&](const IR::StatOrDecl* stmt) { return writeOrReadID(stmt, true); };

    using ID = p4mlir::SSAInfo::ID;

    // bb1
    auto stmtIt = bb1->components.begin();
    // int<16> x$0 = 3;
    ID x0Def = writeID(*stmtIt);
    stmtIt++;
    // int<16> res$0;
    ID res0Def = writeID(*stmtIt);
    *stmtIt++;
    // if (x$0 > 3)
    ID x0Use = readID(*stmtIt);

    // bb2
    stmtIt = bb2->components.begin();
    // x$1 = 2;
    ID x1Def = writeID(*stmtIt);

    // bb3
    stmtIt = bb3->components.begin();
    // x$2 = 1;
    ID x2Def = writeID(*stmtIt);

    // bb4
    stmtIt = bb4->components.begin();
    // x$3 = phi(x$1, x$2)
    ID x3Def = phi.destination.value();
    ID x1Use = phi.sources.at(bb2).value();
    ID x2Use = phi.sources.at(bb3).value();
    // res$1 = x$3;
    ID res1Def = writeID(*stmtIt);
    ID x3Use = readID(*stmtIt);

    EXPECT_EQ(x0Def, x0Use);
    EXPECT_EQ(x1Def, x1Use);
    EXPECT_EQ(x2Def, x2Use);
    EXPECT_EQ(x3Def, x3Use);

    EXPECT_NE(res0Def, res1Def);
    EXPECT_NE(x0Def, x1Def);
    EXPECT_NE(x1Def, x2Def);
    EXPECT_NE(x2Def, x3Def);
}


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
