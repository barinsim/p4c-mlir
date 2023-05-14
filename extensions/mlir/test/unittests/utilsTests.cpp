#include "gtest/gtest.h"

#include <unordered_set>
#include "test/gtest/helpers.h"

#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/toP4/toP4.h"

#include "common.h"
#include "utils.h"

namespace p4mlir::tests {

class MakeFullyQualifiedSymbols : public Test::P4CTest {};

TEST_F(MakeFullyQualifiedSymbols, Test_all_referenceable_constructs) {
    std::string src = P4_SOURCE(R"(
        action foo() {
            return;
        }

        extern bit<16> faz(int<10> arg);

        control Pipe1() {
            action foo() {}
            action bar() {}
            apply {}
        }

        control Pipe2() {
            action foo(int<16> x1) {}
            action baz() {}
            apply {}
        }

        control Pipe3(int<16> arg) {
            apply {}
        }
    )");
    auto output = parseP4ForTests(src);
    ASSERT_TRUE(output && ::errorCount() == 0);
    auto *program = output.ast;

    auto ctxt = createMLIRContext();
    p4mlir::FullyQualifiedSymbols symbols;
    p4mlir::MakeFullyQualifiedSymbols makeSymbols(*ctxt.builder, symbols);
    program->apply(makeSymbols);
    auto strs = symbols.getAllAsStrings();

    std::unordered_set<std::string> syms(strs.begin(), strs.end());
    EXPECT_TRUE(syms.count("foo"));
    EXPECT_TRUE(syms.count("faz"));
    EXPECT_TRUE(syms.count("Pipe1"));
    EXPECT_TRUE(syms.count("Pipe1::foo"));
    EXPECT_TRUE(syms.count("Pipe1::bar"));
    EXPECT_TRUE(syms.count("Pipe2"));
    EXPECT_TRUE(syms.count("Pipe2::foo"));
    EXPECT_TRUE(syms.count("Pipe2::baz"));
    EXPECT_TRUE(syms.count("Pipe3"));
    EXPECT_EQ(syms.size(), 9ULL);
}

class AddRealActionParams : public Test::P4CTest {};

TEST_F(AddRealActionParams, Test_appending_params_arg_with_shadowing) {
    std::string src = P4_SOURCE(R"(
        action foo() {}

        control Pipe2() {
            action bak() { foo(); }
            bit<10> mem1;
            int<20> mem2;
            action foo(in bit<10> x2, int<20> x1) {
                mem1 = 4;
            }
            int<5> mem3;
            bit<5> mem4;
            action baz() {
                int<20> tmp = 3;
                foo(1, tmp);
            }
            action bar(inout int<20> x1, out bit<10> x2) {
                int<20> mem2 = 3;
                if (x1 == mem2) {
                    foo(1, mem2);
                }
            }
            apply {
                int<5> mem4 = 1;
                mem3 = 3 + mem4;
            }
        }
    )");

    auto output = parseP4ForTests(src);
    ASSERT_TRUE(output && ::errorCount() == 0);
    auto* program = output.ast;

    program = program->apply(p4mlir::AddRealActionParams());
    ASSERT_TRUE(program && ::errorCount() == 0);

    EXPECT_EQ(tokenize(P4::toP4(program)), tokenize(
      R"(action foo() {
        }
        control Pipe2() {
            action bak() {
                foo();
            }
            bit<10> __mem1;
            int<20> __mem2;
            action foo(inout bit<10> __mem1, inout int<20> __mem2, in bit<10> x2, int<20> x1) {
                __mem1 = (bit<10>)10w4;
            }
            int<5> __mem3;
            bit<5> __mem4;
            action baz(inout bit<10> __mem1, inout int<20> __mem2, inout int<5> __mem3, inout bit<5> __mem4) {
                int<20> tmp = (int<20>)20s3;
                foo(__mem1, __mem2, (bit<10>)10w1, tmp);
            }
            action bar(inout bit<10> __mem1, inout int<20> __mem2, inout int<5> __mem3, inout bit<5> __mem4, inout int<20> x1, out bit<10> x2) {
                int<20> mem2 = (int<20>)20s3;
                if (x1 == mem2) {
                    foo(__mem1, __mem2, (bit<10>)10w1, mem2);
                }
            }
            apply {
                int<5> mem4 = (int<5>)5s1;
                __mem3 = 5s3 + mem4;
            }
        })"));
}

} // namespace p4mlir::tests
