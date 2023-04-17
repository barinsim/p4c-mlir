// RUN: p4c-mlir-translate %s | FileCheck %s

action bar1(inout int<10> arg1) {
    return;
}

action bar2(out int<10> arg1) {
    return;
}

action bar3(in int<10> arg1, int<10> arg2) {
    return;
}

action foo() {
    int<10> x1 = 2;
    bar1(x1);
    if (x1 == 3) {
        bar2(x1);
    } else {
        bar3(x1, x1);
    }
}

// CHECK: p4.action @bar1(%arg0: !p4.ref<si10>) {
    // CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @bar2(%arg0: !p4.ref<si10>) {
    // CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @bar3(%arg0: si10, %arg1: si10) {
    // CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @foo() {
    // CHECK-NEXT: %0 = p4.constant 2 : si10
    // CHECK-NEXT: %1 = p4.cast(%0) : si10 -> si10
    // CHECK-NEXT: %2 = p4.alloc : !p4.ref<si10>
    // CHECK-NEXT: p4.store(%2, %1) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %3 = p4.alloc : !p4.ref<si10>
    // CHECK-NEXT: %4 = p4.load(%2) : !p4.ref<si10> -> si10
    // CHECK-NEXT: p4.store(%3, %4) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: p4.call @bar1(%3) : (!p4.ref<si10>) -> ()
    // CHECK-NEXT: %5 = p4.load(%3) : !p4.ref<si10> -> si10
    // CHECK-NEXT: p4.store(%2, %5) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %6 = p4.load(%2) : !p4.ref<si10> -> si10
    // CHECK-NEXT: %7 = p4.constant 3 : si64
    // CHECK-NEXT: %8 = p4.cast(%7) : si64 -> si10
    // CHECK-NEXT: %9 = p4.cmp(%6, %8) eq : (si10, si10) -> i1
    // CHECK-NEXT: cf.cond_br %9, ^bb1, ^bb2

        // CHECK-NEXT: ^bb1:  // pred: ^bb0
        // CHECK-NEXT: %10 = p4.alloc : !p4.ref<si10>
        // CHECK-NEXT: p4.call @bar2(%10) : (!p4.ref<si10>) -> ()
        // CHECK-NEXT: %11 = p4.load(%10) : !p4.ref<si10> -> si10
        // CHECK-NEXT: p4.store(%2, %11) : (!p4.ref<si10>, si10) -> ()
        // CHECK-NEXT: cf.br ^bb3

        // CHECK-NEXT: ^bb2:  // pred: ^bb0
        // CHECK-NEXT: %12 = p4.load(%2) : !p4.ref<si10> -> si10
        // CHECK-NEXT: %13 = p4.load(%2) : !p4.ref<si10> -> si10
        // CHECK-NEXT: p4.call @bar3(%12, %13) : (si10, si10) -> ()
        // CHECK-NEXT: cf.br ^bb3

    // CHECK-NEXT: ^bb3:  // 2 preds: ^bb1, ^bb2
    // CHECK-NEXT: p4.return
// CHECK-NEXT: }
