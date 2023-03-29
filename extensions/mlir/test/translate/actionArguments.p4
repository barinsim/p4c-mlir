// RUN: p4c-mlir-translate %s | FileCheck %s

// CHECK-LABEL: module
// CHECK: ^bb0(%arg0: ui16, %arg1: si10, %arg2: ui16):
action foo(in bit<16> arg1, in int<10> arg2, bit<16> arg3) {
    // CHECK: %0 = p4.copy(%arg0) : ui16 -> ui16
    bit<16> x = arg1;
    // CHECK-NEXT: %1 = p4.constant 3 : ui16
    // CHECK-NEXT: %2 = p4.cast(%1) : ui16 -> ui16
    // CHECK-NEXT: %3 = p4.copy(%2) : ui16 -> ui16
    x = 3;
    // CHECK-NEXT: %4 = "p4.cmp"(%arg0, %arg2) {kind = 0 : i32} : (ui16, ui16) -> i1
    // CHECK-NEXT: cf.cond_br %4, ^bb1, ^bb2(%3 : ui16)
    if (arg1 == arg3) {
    // CHECK: ^bb1:
    // CHECK: %5 = p4.constant 1 : ui16
    // CHECK-NEXT: %6 = p4.cast(%5) : ui16 -> ui16
    // CHECK-NEXT: %7 = p4.copy(%6) : ui16 -> ui16
        x = 1;
        // CHECK-NEXT: cf.br ^bb2(%7 : ui16)
    }
    // CHECK: ^bb2(%8: ui16):
    // CHECK: p4.return
    return;
}

