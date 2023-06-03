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

        extern Reg {
            Reg();
            Reg(in bit<10> arg1);
            void extract();
            void extract(int<32> arg1, int<32> arg2);
            void extract(int<32> arg1);
        }

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
    p4mlir::MakeFullyQualifiedSymbols makeSymbols(*ctxt.builder, symbols, output.typeMap);
    program->apply(makeSymbols);
    auto strs = symbols.getAllAsStrings();

    std::unordered_set<std::string> syms(strs.begin(), strs.end());
    EXPECT_TRUE(syms.count("foo"));
    EXPECT_TRUE(syms.count("faz_1"));
    EXPECT_TRUE(syms.count("Pipe1"));
    EXPECT_TRUE(syms.count("Pipe1::foo"));
    EXPECT_TRUE(syms.count("Pipe1::bar"));
    EXPECT_TRUE(syms.count("Pipe2"));
    EXPECT_TRUE(syms.count("Pipe2::foo"));
    EXPECT_TRUE(syms.count("Pipe2::baz"));
    EXPECT_TRUE(syms.count("Pipe3"));
    EXPECT_TRUE(syms.count("Reg"));
    EXPECT_TRUE(syms.count("Reg::Reg_0"));
    EXPECT_TRUE(syms.count("Reg::Reg_1"));
    EXPECT_TRUE(syms.count("Reg::extract_0"));
    EXPECT_TRUE(syms.count("Reg::extract_2"));
    EXPECT_TRUE(syms.count("Reg::extract_1"));
    EXPECT_EQ(syms.size(), 15ULL);
}

} // namespace p4mlir::tests
