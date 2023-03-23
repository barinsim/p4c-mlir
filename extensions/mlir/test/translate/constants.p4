// RUN: p4c-mlir-translate %s | FileCheck %s
// RUN: p4c-mlir-translate %s | FileCheck %s --check-prefix CHECK-SYMS

// CHECK-LABEL: module
// CHECK: p4.action
action foo() {
    // CHECK-NEXT: {{%[0-9]+}} = p4.constant -5 : si10
    int<10> x = -5;
    // CHECK-NEXT: p4.return
    return;
}

// CHECK: p4.action
action bar() {
    // CHECK-NEXT: {{%[0-9]+}} = p4.constant 19 : ui100
    bit<100> x = 19;
    // CHECK-NEXT: p4.return
    return;
}

// CHECK-SYMS: foo
// CHECK-SYMS: bar

