// RUN: p4c-mlir-translate %s | FileCheck %s

control Pipe(bit<10> arg1, in int<16> arg2) {
    action foo() {
        int<16> x1 = 3;
        x1 = 4;
    }
    action bar() {
        bit<10> x1 = 2;
        x1 = arg1;
    }
    apply {
        bit<10> x1 = arg1;
        int<16> x2 = 5;
        x2 = arg2;
        if (arg2 == 3) {
            x2 = 3;
        }
    }
}

// CHECK-LABEL: module

// CHECK: p4.control @Pipe(%arg0: ui10, %arg1: si16)
// CHECK: p4.action @foo()
// CHECK: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant 3 : si16
// CHECK-NEXT: %2 = p4.cast(%1) : si16 -> si16
// CHECK-NEXT: %3 = p4.copy(%2) : si16 -> si16
// CHECK-NEXT: %4 = p4.constant 4 : si16
// CHECK-NEXT: %5 = p4.cast(%4) : si16 -> si16
// CHECK-NEXT: %6 = p4.copy(%5) : si16 -> si16
// CHECK-NEXT: p4.return

// CHECK: p4.action @bar()
// CHECK: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK: %1 = p4.constant 2 : ui10
// CHECK-NEXT: %2 = p4.cast(%1) : ui10 -> ui10
// CHECK-NEXT: %3 = p4.copy(%2) : ui10 -> ui10
// CHECK-NEXT: %4 = p4.copy(%arg0) : ui10 -> ui10
// CHECK-NEXT: p4.return

// CHECK: p4.apply
// CHECK: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK: %1 = p4.copy(%arg0) : ui10 -> ui10
// CHECK-NEXT: %2 = p4.constant 5 : si16
// CHECK-NEXT: %3 = p4.cast(%2) : si16 -> si16
// CHECK-NEXT: %4 = p4.copy(%3) : si16 -> si16
// CHECK-NEXT: %5 = p4.copy(%arg1) : si16 -> si16
// CHECK-NEXT: %6 = p4.constant 3 : si64
// CHECK-NEXT: %7 = p4.cast(%6) : si64 -> si16
// CHECK-NEXT: %8 = p4.cmp(%arg1, %7) eq : (si16, si16) -> i1
// CHECK-NEXT: cf.cond_br %8, ^bb1, ^bb2(%5 : si16)

// CHECK: ^bb1:
// CHECK-NEXT: %9 = p4.constant 3 : si16
// CHECK-NEXT: %10 = p4.cast(%9) : si16 -> si16
// CHECK-NEXT: %11 = p4.copy(%10) : si16 -> si16
// CHECK-NEXT: cf.br ^bb2(%11 : si16)

// CHECK: ^bb2(%12: si16):
// CHECK: p4.return
