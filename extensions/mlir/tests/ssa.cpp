#include "gtest/gtest.h"

#include "test/gtest/helpers.h"
#include "frontends/common/parseInput.h"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "common.h"
#include "../cfgBuilder.h"
#include "../domTree.h"
#include "../ssa.h"


namespace p4mlir::tests {


class SSAInfo : public Test::P4CTest { };


TEST_F(SSAInfo, Test_action_with_control_flow) {
    std::string src = P4_SOURCE(R"(
        header Header {
            bit<8> f1;
            bit<8> f2;
            bit<8> f3;
            bit<8> f4;
            bit<8> f5;
            bit<8> f6;
            bit<8> f7;
        }
        control Pipe<H>(inout H headers);
        package Pipeline<H>(Pipe<H> p);
        control TopPipe(inout Header hdr) {
            int<16> label;
            action foo() {
                // bb1
                label = 1;
                if (hdr.f2 == hdr.f3) {
                    // bb2
                    label = 2;
                    hdr.f1 = 3;
                    if (hdr.f2 == hdr.f2) {
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
            }
            apply {

            }
        }
        Pipeline(TopPipe()) main;
    )");
    auto* program = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(program && ::errorCount() == 0);

    // auto b = new p4mlir::CFGBuilder;
    // program->apply(*b);
    // auto cfg = b->getCFG();

    auto* refMap = new P4::ReferenceMap();
    auto* typeMap = new P4::TypeMap();
    program->apply(P4::ResolveReferences(refMap));
    program->apply(P4::TypeInference(refMap, typeMap, false, true));
    ASSERT_TRUE(program != nullptr && ::errorCount() == 0);

    // auto* cfgFoo = getByName(cfg, "foo");

    // const p4mlir::DomTree* domTree = p4mlir::DomTree::fromEntryBlock(cfgFoo);
    // const p4mlir::SSAInfo* ssa = p4mlir::SSAInfo::create(cfgFoo, domTree, refMap);
    // ASSERT_TRUE(ssa);

    // p4mlir::BasicBlock* bb1 = getByStmtString(cfgFoo, "label = 1;");
    // p4mlir::BasicBlock* bb2 = getByStmtString(cfgFoo, "label = 2;");
    // p4mlir::BasicBlock* bb3 = getByStmtString(cfgFoo, "label = 3;");
    // p4mlir::BasicBlock* bb4 = getByStmtString(cfgFoo, "label = 4;");
    // p4mlir::BasicBlock* bb5 = getByStmtString(cfgFoo, "label = 5;");
    // p4mlir::BasicBlock* bb6 = getByStmtString(cfgFoo, "label = 6;");

    auto* gatherRefs = new p4mlir::GatherSSAReferences(typeMap, refMap);
    program->apply(*gatherRefs);

    auto refs = gatherRefs->getSSARefs();
    for (auto& r : refs) {
        std::cout << r.toString() << std::endl;
    }

    /*std::string src = P4_SOURCE(R"(
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
            return hdr.f1, hdr.f2, hdr.f6
        }
    )");*/

        // need top sort on call graph!!!
        // must know all writes and reads including called actions
        // from each statement i need writes and reads (member, pathexpression, )
        // extern objects cannot be ssa, they can be invalidated through use of another extern method/function (unless annotated with noSideEffect)
        // 1. analyze/tranform functions - we need to know what values they define, use
        //      - transform function prototypes
        //      - transform all call sites
        /*std::string src = P4_SOURCE(R"(
        action foo() {
            // bb1
            label-1 = 1;
            if (hdr.f2-1 == hdr.f3-1 + 1) {
                // bb2
                label-2 = 2;
                hdr.f1$2 = 3;
                if (hdr.f2$1 == hdr.f2$1 + 1) {
                    // bb3
                    label$4 = 3;
                    hdr.f2$2 = 3;
                } else {
                    // bb4
                    label$5 = 4;
                    hdr.f2$3 = 6;
                }
                // bb5
                // phis
                label$6 = 5;
                hdr.f6$2 = hdr.f1$2;
            }
            // bb6
            label$7 = 6;
                                __ __ __ __ = test.apply(__ __ __ __ __);
            return hdr.f1, hdr.f2, hdr.f6
        }
    )");*/
}


} // namespace p4mlir::tests
