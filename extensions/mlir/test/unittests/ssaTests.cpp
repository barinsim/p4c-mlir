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

namespace {
    struct Output {
        const CFGInfo& cfgInfo;
        const Allocation& allocation;
        const SSAInfo& ssaInfo;
        const IR::P4Program* program = nullptr;
        const P4::TypeMap* typeMap = nullptr;
        const P4::ReferenceMap* refMap = nullptr;

        Output(const CFGInfo &cfgInfo_, const Allocation &allocation_, const SSAInfo &ssaInfo_,
               const IR::P4Program *program_, const P4::TypeMap *typeMap_,
               const P4::ReferenceMap *refMap_)
            : cfgInfo(cfgInfo_),
              allocation(allocation_),
              ssaInfo(ssaInfo_),
              program(program_),
              typeMap(typeMap_),
              refMap(refMap_) {
            CHECK_NULL(program, typeMap, refMap);
        }
    };

    Output runTestPasses(const std::string& p4src) {
        ParseOutput po = parseP4ForTests(p4src);
        if (!po || ::errorCount() > 0) {
            BUG_CHECK(false, "");
        }

        auto* program = po.ast;
        auto* typeMap = po.typeMap;
        auto* refMap = po.refMap;

        auto* cfgInfo = new CFGInfo();
        program->apply(p4mlir::MakeCFGInfo(*cfgInfo));
        if (::errorCount() > 0) {
            BUG_CHECK(false, "");
        }

        auto* allocation = new p4mlir::Allocation();
        program->apply(p4mlir::AllocateVariables(refMap, typeMap, *allocation));
        if (::errorCount() > 0) {
            BUG_CHECK(false, "");
        }

        auto* ssaInfo = new p4mlir::SSAInfo();
        program->apply(p4mlir::MakeSSAInfo(*ssaInfo, *cfgInfo, *allocation, refMap, typeMap));
        if (::errorCount() > 0) {
            BUG_CHECK(false, "");
        }

        return Output(*cfgInfo, *allocation, *ssaInfo, program, typeMap, refMap);
    }
}

class SSAInfo : public Test::P4CTest {};

TEST_F(SSAInfo, Test_ssa_conversion_for_simple_action_1) {
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
    Output out = runTestPasses(src);
    auto& cfgInfo = out.cfgInfo;
    auto& ssaInfo = out.ssaInfo;
    auto& allocation = out.allocation;
    auto* typeMap = out.typeMap;
    auto* refMap = out.refMap;

    ASSERT_EQ(cfgInfo.size(), (std::size_t)1);
    auto foo = getByName(cfgInfo, "foo");

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
    p4mlir::PhiInfo phi = phiInfo.begin()->second;

    EXPECT_TRUE(phi.destination.has_value());
    EXPECT_TRUE(phi.sources.at(bb2).has_value());
    EXPECT_TRUE(phi.sources.at(bb3).has_value());

    // This relies on a single def/use within a statement.
    // Ignores stack allocated vars
    auto writeOrReadID = [&](const IR::StatOrDecl* stmt, bool reads) {
        p4mlir::GatherSSAReferences refs(typeMap, refMap, allocation);
        stmt->apply(refs);
        std::vector<p4mlir::RefInfo> infos;
        if (reads) {
            infos = refs.getReads();
        } else {
            infos = refs.getWrites();
        }
        EXPECT_EQ(infos.size(), (std::size_t)1);
        auto info = infos.front();
        return ssaInfo.getID(info.ref);
    };
    auto writeID = [&](const IR::StatOrDecl* stmt) { return writeOrReadID(stmt, false); };
    auto readID = [&](const IR::StatOrDecl* stmt) { return writeOrReadID(stmt, true); };

    using ID = p4mlir::ID;

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

TEST_F(SSAInfo, Test_ssa_conversion_for_simple_action_2) {
    std::string src = P4_SOURCE(R"(
        action foo() {
            // bb1
            int<16> x1 = 1;
            if (x1 == 2) {
                // bb2
                int<16> x2 = 3;
                x1 = 3;
            } else {
                // bb3
                int<16> x3 = 4;
                x1 = 4;
            }
            // bb4
            int<16> x4 = 5;
            return;
        }
    )");
    Output out = runTestPasses(src);
    auto& cfgInfo = out.cfgInfo;
    auto& ssaInfo = out.ssaInfo;
    auto& allocation = out.allocation;
    auto* typeMap = out.typeMap;
    auto* refMap = out.refMap;

    ASSERT_EQ(cfgInfo.size(), (std::size_t)1);
    auto cfgFoo = getByName(cfgInfo, "foo");

    auto* bb1 = getByStmtString(cfgFoo, "int<16> x1 = (int<16>)16s1;");
    auto* bb2 = getByStmtString(cfgFoo, "int<16> x2 = (int<16>)16s3;");
    auto* bb3 = getByStmtString(cfgFoo, "int<16> x3 = (int<16>)16s4;");
    auto* bb4 = getByStmtString(cfgFoo, "int<16> x4 = (int<16>)16s5;");

    auto names = [](auto decls) {
        std::unordered_set<cstring> res;
        for (auto[d, p] : decls) {
            res.insert(d->getName().name);
        }
        return res;
    };

    using unordered = std::unordered_set<cstring>;
    EXPECT_EQ(names(ssaInfo.getPhiInfo(bb1)), unordered({}));
    EXPECT_EQ(names(ssaInfo.getPhiInfo(bb2)), unordered({}));
    EXPECT_EQ(names(ssaInfo.getPhiInfo(bb3)), unordered({}));
    EXPECT_EQ(names(ssaInfo.getPhiInfo(bb4)), unordered({"x1"}));
}

TEST_F(SSAInfo, Correctly_detect_SSA_reads_and_writes_after_allocation) {
    std::string src = P4_SOURCE(R"(
        extern int<16> bar1(inout int<16> x1, in int<16> x2);
        extern int<16> bar2(out int<16> x3);
        extern int<16> bar3(in int<16> x4, in int<16> x5, in int<16> x6);
        action foo(in int<16> f7, inout int<16> f8, out int<16> f9) {
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
            f8 = 2;
            f9 = 1;
            bar3(f8, f8, f8);
            return;
        }
    )");
    Output out = runTestPasses(src);
    auto& cfgInfo = out.cfgInfo;
    auto& ssaInfo = out.ssaInfo;
    auto& allocation = out.allocation;
    auto* typeMap = out.typeMap;
    auto* refMap = out.refMap;

    ASSERT_EQ(cfgInfo.size(), (std::size_t)1);
    auto fooCFG = cfgInfo.begin()->second;

    auto ssaVars = allocation.getAllOf(p4mlir::AllocType::REG);
    using unordered = std::unordered_set<cstring>;
    EXPECT_EQ(names(ssaVars), unordered({"f3", "f5", "f6", "f7", "x2", "x4", "x5", "x6"}));

    auto getRefs = [&](auto* stmt, bool reads) {
        p4mlir::GatherSSAReferences refs(typeMap, refMap, allocation);
        stmt->apply(refs);
        std::vector<const IR::IDeclaration*> decls;
        auto infos = refs.getWrites();
        if (reads) {
            infos = refs.getReads();
        }
        std::transform(infos.begin(), infos.end(), std::back_inserter(decls),
                       [](auto& info) { return info.decl; });
        return names(decls);
    };
    auto writes = [&](auto *stmt) { return getRefs(stmt, false); };
    auto reads = [&](auto *stmt) { return getRefs(stmt, true); };

    auto stmtIt = fooCFG.getEntry()->components.begin();

    // int<16> f1 = 3;
    EXPECT_EQ(writes(*stmtIt), unordered({}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // int<16> f2 = 3;
    EXPECT_EQ(writes(*stmtIt), unordered({}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // int<16> f3 = 3;
    EXPECT_EQ(writes(*stmtIt), unordered({"f3"}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // int<16> f4 = 3;
    EXPECT_EQ(writes(*stmtIt), unordered({}));
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
    // f8 = 2;
    EXPECT_EQ(writes(*stmtIt), unordered({}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // f9 = 1;
    EXPECT_EQ(writes(*stmtIt), unordered({}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;
    // bar1(f8, f8, f8);
    EXPECT_EQ(writes(*stmtIt), unordered({}));
    EXPECT_EQ(reads(*stmtIt), unordered({}));
    stmtIt++;

    EXPECT_EQ(stmtIt, fooCFG.getEntry()->components.end());
}

TEST_F(SSAInfo, Test_SSA_calculation_of_parameters) {
    std::string src = P4_SOURCE(R"(
        extern int<16> bar1(inout int<16> x1, in int<16> x2);
        extern int<16> bar2(out int<16> x3);

        control Pipe(in int<16> arg1, out int<16> arg2, inout int<16> arg3)(bool arg4) {
            int<16> arg5;
            action foo(in int<16> arg6, out int<16> arg7, inout int<16> arg8, bit<10> arg9) {
                return;
            }
            apply {
            }
        }
    )");
    Output out = runTestPasses(src);
    auto& ssaInfo = out.ssaInfo;
    auto& allocation = out.allocation;

    // None and In direction parameters are REG allocated
    auto ssaVars = allocation.getAllOf(p4mlir::AllocType::REG);
    using unordered = std::unordered_set<cstring>;
    EXPECT_EQ(names(ssaVars), unordered({"x2", "arg1", "arg4", "arg6", "arg9"}));

    // All REG allocated parameters have SSA number assigned, except extern parameters (they cannot
    // be referenced within the program)
    for(auto* decl : ssaVars) {
        if (decl->getName() == "x2") {
            continue;
        }
        EXPECT_NO_THROW(ssaInfo.getID(decl));
    }
}


} // namespace p4mlir::tests
