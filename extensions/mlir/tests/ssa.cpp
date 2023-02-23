#include "gtest/gtest.h"

#include "test/gtest/helpers.h"
#include "frontends/common/parseInput.h"
#include "common.h"
#include "../cfgBuilder.h"
#include "../domTree.h"
#include "../ssa.h"


namespace p4mlir::tests {


class SSAInfo : public Test::P4CTest { };


TEST_F(SSAInfo, Test_action_with_control_flow) {
    std::string src = P4_SOURCE(R"(
        action foo() {
            // bb1
            label = 1;
            if (hdr.f2 == hdr.f3 + 1) {
                // bb2
                label = 2;
                hdr.f1 = 3;
                if (hdr.f2 == hdr.f2 + 1) {
                    // bb3
                    label = 3;
                    hdr.f2 = 3;
                } else {
                    // bb4
                    label = 4;
                    hdr.f2 = 6;
                }
                // bb5
                label = 5;
                hdr.f6 = hdr.f1;
            }
            // bb6
            label = 6;
            test.apply();
        }
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto cfg = b->getCFG();

    auto* refMap = new P4::ReferenceMap();
    program->apply(P4::ResolveReferences(refMap));

    const p4mlir::DomTree* domTree = p4mlir::DomTree::fromEntryBlock(cfgFoo);
    const p4mlir::SSAInfo* ssa = p4mlir::SSAInfo::create(cfgFoo, domTree, refMap);
    ASSERT_TRUE(ssa);

    auto* cfgFoo = getByName(cfg, "foo");

    p4mlir::BasicBlock* bb1 = getByStmtString(cfgFoo, "label = 1;");
    p4mlir::BasicBlock* bb2 = getByStmtString(cfgFoo, "label = 2;");
    p4mlir::BasicBlock* bb3 = getByStmtString(cfgFoo, "label = 3;");
    p4mlir::BasicBlock* bb4 = getByStmtString(cfgFoo, "label = 4;");
    p4mlir::BasicBlock* bb5 = getByStmtString(cfgFoo, "label = 5;");
    p4mlir::BasicBlock* bb6 = getByStmtString(cfgFoo, "label = 6;");


    EXPECT_EQ(ssa->)
}


} // namespace p4mlir::tests
