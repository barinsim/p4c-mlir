// RUN: p4c-mlir-translate %s | FileCheck %s

action foo() {
    int<16> x1 = 1;
    int<16> x2 = 3;
    int<16> x3 = 4;
    if (x1 == 2) {
        x1 = 2;
        x2 = 4;
    } else {
        x3 = 5;
        x1 = 7;
    }
    if (x1 == x2) {
        int<16> x4 = x3;
        x1 = x3;
    }
    x2 = 1;
    return;
}

// CHECK-LABEL: module
// CHECK: %0 = p4.constant 1 : si16
// CHECK-NEXT: %1 = p4.cast(%0) : si16 -> si16
// CHECK-NEXT: %2 = p4.copy(%1) : si16 -> si16
// CHECK-NEXT: %3 = p4.constant 3 : si16
// CHECK-NEXT: %4 = p4.cast(%3) : si16 -> si16
// CHECK-NEXT: %5 = p4.copy(%4) : si16 -> si16
// CHECK-NEXT: %6 = p4.constant 4 : si16
// CHECK-NEXT: %7 = p4.cast(%6) : si16 -> si16
// CHECK-NEXT: %8 = p4.copy(%7) : si16 -> si16
// CHECK-NEXT: %9 = p4.constant 2 : si64
// CHECK-NEXT: %10 = p4.cast(%9) : si64 -> si16
// CHECK-NEXT: %11 = p4.cmp(%2, %10) eq : (si16, si16) -> i1
// CHECK-NEXT: cf.cond_br %11, ^bb1, ^bb2

    // CHECK: ^bb1:
    // CHECK-NEXT: %12 = p4.constant 2 : si16
    // CHECK-NEXT: %13 = p4.cast(%12) : si16 -> si16
    // CHECK-NEXT: %14 = p4.copy(%13) : si16 -> si16
    // CHECK-NEXT: %15 = p4.constant 4 : si16
    // CHECK-NEXT: %16 = p4.cast(%15) : si16 -> si16
    // CHECK-NEXT: %17 = p4.copy(%16) : si16 -> si16
    // CHECK-NEXT: cf.br ^bb3(%14, %17, %8 : si16, si16, si16)

    // CHECK: ^bb2:
    // CHECK-NEXT: %18 = p4.constant 5 : si16
    // CHECK-NEXT: %19 = p4.cast(%18) : si16 -> si16
    // CHECK-NEXT: %20 = p4.copy(%19) : si16 -> si16
    // CHECK-NEXT: %21 = p4.constant 7 : si16
    // CHECK-NEXT: %22 = p4.cast(%21) : si16 -> si16
    // CHECK-NEXT: %23 = p4.copy(%22) : si16 -> si16
    // CHECK-NEXT: cf.br ^bb3(%23, %5, %20 : si16, si16, si16)

// CHECK: ^bb3(%24: si16, %25: si16, %26: si16):
// CHECK-NEXT: %27 = p4.cmp(%24, %25) eq : (si16, si16) -> i1
// CHECK-NEXT: cf.cond_br %27, ^bb4, ^bb5(%24 : si16)

    // CHECK: ^bb4:
    // CHECK-NEXT: %28 = p4.copy(%26) : si16 -> si16
    // CHECK-NEXT: %29 = p4.copy(%26) : si16 -> si16
    // CHECK-NEXT: cf.br ^bb5(%29 : si16)

// CHECK: ^bb5(%30: si16):
// CHECK-NEXT: %31 = p4.constant 1 : si16
// CHECK-NEXT: %32 = p4.cast(%31) : si16 -> si16
// CHECK-NEXT: %33 = p4.copy(%32) : si16 -> si16
// CHECK-NEXT: p4.return




