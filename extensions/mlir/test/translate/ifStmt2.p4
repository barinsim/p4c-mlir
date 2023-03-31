// RUN: p4c-mlir-translate %s | FileCheck %s

action foo() {
    int<16> x1 = 1;
    if (x1 == 2) {
        int<16> x2 = 3;
    } else {
        int<16> x3 = 4;
    }
    int<16> x4 = 5;
    return;
}

// CHECK-LABEL: module

// CHECK: %0 = p4.constant 1 : si16
// CHECK-NEXT: %1 = p4.cast(%0) : si16 -> si16
// CHECK-NEXT: %2 = p4.copy(%1) : si16 -> si16
// CHECK-NEXT: %3 = p4.constant 2 : si64
// CHECK-NEXT: %4 = p4.cast(%3) : si64 -> si16
// CHECK-NEXT: %5 = "p4.cmp"(%2, %4) {kind = 0 : i32} : (si16, si16) -> i1
// CHECK-NEXT: cf.cond_br %5, ^bb1, ^bb2

    // CHECK: ^bb1:
    // CHECK-NEXT: %6 = p4.constant 3 : si16
    // CHECK-NEXT: %7 = p4.cast(%6) : si16 -> si16
    // CHECK-NEXT: %8 = p4.copy(%7) : si16 -> si16
    // CHECK-NEXT: cf.br ^bb3

    // CHECK: ^bb2:
    // CHECK-NEXT: %9 = p4.constant 4 : si16
    // CHECK-NEXT: %10 = p4.cast(%9) : si16 -> si16
    // CHECK-NEXT: %11 = p4.copy(%10) : si16 -> si16
    // CHECK-NEXT: cf.br ^bb3

// CHECK-DAG: ^bb3:
// CHECK-NEXT: %12 = p4.constant 5 : si16
// CHECK-NEXT: %13 = p4.cast(%12) : si16 -> si16
// CHECK-NEXT: %14 = p4.copy(%13) : si16 -> si16
// CHECK-NEXT: p4.return
