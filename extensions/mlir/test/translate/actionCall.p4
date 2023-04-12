// RUN: p4c-mlir-translate %s | FileCheck %s

action foo(in int<16> arg2, bit<10> arg1) {
    int<16> x1 = 3;
    x1 = arg2;
    bit<10> x2 = arg1;
}

action bar() {
    int<16> x1 = 2;
    return;
}

action baz(in int<16> arg1) {
    foo(arg1, 7);
    bit<10> x1 = 5;
    foo(arg1, x1);
    foo(4, 2);
    bar();
    return;
}

// CHECK-LABEL: module {
    // CHECK-NEXT: p4.action @foo(%arg0: si16, %arg1: ui10) {
    // CHECK-NEXT: %0 = p4.constant 3 : si16
    // CHECK-NEXT: %1 = p4.cast(%0) : si16 -> si16
    // CHECK-NEXT: %2 = p4.copy(%1) : si16 -> si16
    // CHECK-NEXT: %3 = p4.copy(%arg0) : si16 -> si16
    // CHECK-NEXT: %4 = p4.copy(%arg1) : ui10 -> ui10
    // CHECK-NEXT: p4.return
  // CHECK-NEXT: }
  // CHECK-NEXT: p4.action @bar() {
    // CHECK-NEXT: %0 = p4.constant 2 : si16
    // CHECK-NEXT: %1 = p4.cast(%0) : si16 -> si16
    // CHECK-NEXT: %2 = p4.copy(%1) : si16 -> si16
    // CHECK-NEXT: p4.return
  // CHECK-NEXT: }
  // CHECK-NEXT: p4.action @baz(%arg0: si16) {
    // CHECK-NEXT: %0 = p4.constant 7 : ui10
    // CHECK-NEXT: p4.call @foo(%arg0, %0) : (si16, ui10) -> ()
    // CHECK-NEXT: %1 = p4.constant 5 : ui10
    // CHECK-NEXT: %2 = p4.cast(%1) : ui10 -> ui10
    // CHECK-NEXT: %3 = p4.copy(%2) : ui10 -> ui10
    // CHECK-NEXT: p4.call @foo(%arg0, %3) : (si16, ui10) -> ()
    // CHECK-NEXT: %4 = p4.constant 4 : si16
    // CHECK-NEXT: %5 = p4.cast(%4) : si16 -> si16
    // CHECK-NEXT: %6 = p4.constant 2 : ui10
    // CHECK-NEXT: p4.call @foo(%5, %6) : (si16, ui10) -> ()
    // CHECK-NEXT: p4.call @bar() : () -> ()
    // CHECK-NEXT: p4.return
  // CHECK-NEXT: }
// CHECK-NEXT: }
