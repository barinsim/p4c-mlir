// RUN: p4c-mlir-translate %s | FileCheck %s

action bit_comparisons() {
    bool res;

    bit<10> lhs = 1;
    bit<10> rhs = 2;

    // CHECK: %7 = p4.cmp(%3, %6) eq : (ui10, ui10) -> i1
    res = lhs == rhs;
    // CHECK: %9 = p4.cmp(%3, %6) ne : (ui10, ui10) -> i1
    res = lhs != rhs;
    // CHECK: %11 = p4.cmp(%3, %6) lt : (ui10, ui10) -> i1
    res = lhs < rhs;
    // CHECK: %13 = p4.cmp(%3, %6) le : (ui10, ui10) -> i1
    res = lhs <= rhs;
    // CHECK: %15 = p4.cmp(%3, %6) gt : (ui10, ui10) -> i1
    res = lhs > rhs;
    // CHECK: %17 = p4.cmp(%3, %6) ge : (ui10, ui10) -> i1
    res = lhs >= rhs;
}

action int_comparisons() {
    bool res;

    int<5> lhs = 1;
    int<5> rhs = 2;

    // CHECK: %7 = p4.cmp(%3, %6) eq : (si5, si5) -> i1
    res = lhs == rhs;
    // CHECK: %9 = p4.cmp(%3, %6) ne : (si5, si5) -> i1
    res = lhs != rhs;
    // CHECK: %11 = p4.cmp(%3, %6) lt : (si5, si5) -> i1
    res = lhs < rhs;
    // CHECK: %13 = p4.cmp(%3, %6) le : (si5, si5) -> i1
    res = lhs <= rhs;
    // CHECK: %15 = p4.cmp(%3, %6) gt : (si5, si5) -> i1
    res = lhs > rhs;
    // CHECK: %17 = p4.cmp(%3, %6) ge : (si5, si5) -> i1
    res = lhs >= rhs;
}

action bool_comparisons() {
    bool res;

    bool lhs = true;
    bool rhs = false;

    // CHECK: %5 = p4.cmp(%2, %4) eq : (i1, i1) -> i1
    res = lhs == rhs;
    // CHECK: %7 = p4.cmp(%2, %4) ne : (i1, i1) -> i1
    res = lhs != rhs;

    // CHECK: %9 = p4.constant true
    // CHECK: %10 = p4.constant false
    // CHECK: %11 = p4.cmp(%9, %10) eq : (i1, i1) -> i1
    res = true == false;

    // CHECK: %13 = p4.constant true
    // CHECK: %14 = p4.constant false
    // CHECK: %15 = p4.cmp(%13, %14) ne : (i1, i1) -> i1
    res = true != false;
}

// TODO: varbits, arbitrary precision ints
