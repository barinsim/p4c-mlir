// RUN: p4c-mlir-translate %s | FileCheck %s

// CHECK-LABEL: module
action foo() {
    // CHECK-NEXT: [[reg1:%[0-9]+]] = p4.constant 1 : si16
    int<16> x1 = 1;
    // CHECK-NEXT: p4.eq([[reg1]]) : (si16, si16) -> bool
    // CHECK-NEXT: [[reg2:%[0-9]+]] = p4.constant 2 : si16
    // CHECK-NEXT: [[reg3:%[0-9]+]] = p4.eq([[reg1]], [[reg2]]) : (si16, si16) -> bool
    // CHECK-NEXT: p4.cond [[reg3]] [[bb1:bb[0-9]+]]^ [[bb2:bb[0-9]+]]^
    if (x1 == 2) {
        // CHECK: = p4.constant 3 : si16
        int<16> x2 = 3;
        // CHECK-NEXT: cf.br
    } else {
        // CHECK: = p4.constant 4 : si16
        int<16> x3 = 4;
        // CHECK-NEXT: cf.br
    }
    // CHECK-NEXT: = p4.constant 5 : si16
    int<16> x4 = 5;
    // CHECK-NEXT: = p4.return
    return;
}




