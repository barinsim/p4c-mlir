// RUN: p4c-mlir-translate %s | FileCheck %s

// CHECK-LABEL: module
// CHECK: p4.action @foo()
action foo() {
    // CHECK: [[R1:%[0-9]+]] = p4.constant -5 : si10
    // CHECK-NEXT: [[R2:%[0-9]+]] = p4.cast([[R1]]) : si10 -> si10
    // CHECK-NEXT: = p4.copy([[R2]]) : si10 -> si10
    int<10> x = -5;
    // CHECK-NEXT: p4.return
    return;
}

// CHECK: p4.action @bar()
action bar() {
    // CHECK: [[R3:%[0-9]+]] = p4.constant 19 : ui100
    // CHECK-NEXT: [[R4:%[0-9]+]] = p4.cast([[R3]]) : ui100 -> ui100
    // CHECK-NEXT: = p4.copy([[R4]]) : ui100 -> ui100
    bit<100> x = 19;
    // CHECK-NEXT: p4.return
    return;
}
