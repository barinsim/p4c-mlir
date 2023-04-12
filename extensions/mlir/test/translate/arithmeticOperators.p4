// RUN: p4c-mlir-translate %s | FileCheck %s

action bit_operators() {
    bit<10> res;

    bit<10> lhs = 1;
    bit<10> rhs = 2;

    // CHECK: %7 = p4.add(%3, %6) : (ui10, ui10) -> ui10
    res = lhs + rhs;
    // CHECK: %9 = p4.sub(%3, %6) : (ui10, ui10) -> ui10
    res = lhs - rhs;
    // CHECK: %11 = p4.mul(%3, %6) : (ui10, ui10) -> ui10
    res = lhs * rhs;

    // CHECK: %13 = p4.constant 3 : ui10
    // CHECK: %14 = p4.mul(%3, %13) : (ui10, ui10) -> ui10
    res = lhs * 3;

    // CHECK: %16 = p4.constant 2 : ui10
    // CHECK: %17 = p4.mul(%16, %6) : (ui10, ui10) -> ui10
    res = 2 * rhs;

    // CHECK: %19 = p4.mul(%6, %6) : (ui10, ui10) -> ui10
    res = rhs * rhs;

    // CHECK: %21 = p4.constant 3 : ui10
    // CHECK: %22 = p4.mul(%6, %21) : (ui10, ui10) -> ui10
    // CHECK: %23 = p4.add(%3, %22) : (ui10, ui10) -> ui10
    // CHECK: %24 = p4.constant 2 : ui10
    // CHECK: %25 = p4.sub(%24, %6) : (ui10, ui10) -> ui10
    // CHECK: %26 = p4.add(%23, %25) : (ui10, ui10) -> ui10
    res = lhs + rhs * 3 + (2 - rhs);
}

action int_operators() {
    int<10> res;

    int<10> lhs = 1;
    int<10> rhs = 2;

    // CHECK: %7 = p4.add(%3, %6) : (si10, si10) -> si10
    res = lhs + rhs;
    // CHECK: %9 = p4.sub(%3, %6) : (si10, si10) -> si10
    res = lhs - rhs;
    // CHECK: %11 = p4.mul(%3, %6) : (si10, si10) -> si10
    res = lhs * rhs;

    // CHECK: %13 = p4.constant 3 : si10
    // CHECK: %14 = p4.mul(%3, %13) : (si10, si10) -> si10
    res = lhs * 3;

    // CHECK: %16 = p4.constant 2 : si10
    // CHECK: %17 = p4.mul(%16, %6) : (si10, si10) -> si10
    res = 2 * rhs;

    // CHECK: %19 = p4.mul(%6, %6) : (si10, si10) -> si10
    res = rhs * rhs;

    // CHECK: %21 = p4.constant 3 : si10
    // CHECK: %22 = p4.mul(%6, %21) : (si10, si10) -> si10
    // CHECK: %23 = p4.add(%3, %22) : (si10, si10) -> si10
    // CHECK: %24 = p4.constant 2 : si10
    // CHECK: %25 = p4.sub(%24, %6) : (si10, si10) -> si10
    // CHECK: %26 = p4.add(%23, %25) : (si10, si10) -> si10
    res = lhs + rhs * 3 + (2 - rhs);
}

// TODO: unary operators, varbits, arbitrary precision ints
