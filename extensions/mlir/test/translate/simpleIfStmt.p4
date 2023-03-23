// RUN: p4c-mlir-translate %s | FileCheck %s

// CHECK-LABEL: module
action foo() {
    // CHECK: [[REG1:%[0-9]+]] = p4.constant true
    // CHECK-NEXT: cf.cond_br [[REG1]], ^[[BB1:bb[0-9]+]], ^[[BB1]]
    if (true) {

    } else {

    }
    // CHECK-NEXT: ^[[BB1]]
    // CHECK-NEXT: p4.return
    return;
}
