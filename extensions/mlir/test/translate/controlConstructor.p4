// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
}

action foo(bool flag) {}

control Pipe(bit<10> arg1, in int<16> arg2, out int<16> arg3, inout int<16> arg4)
            (bool ctr_arg1, int<16> ctr_arg2, MyHeader ctr_arg3) {
    action bar(in int<16> a1) {
        int<16> x1;
        if (ctr_arg3.f1 >= 2) {
            x1 = ctr_arg2 + a1;
        } else {
            x1 = ctr_arg3.f1;
        }
        arg4 = x1;
        foo(ctr_arg1);
    }

    apply {
        MyHeader x1 = ctr_arg3;
        if (ctr_arg1) {
            arg3 = ctr_arg2 + ctr_arg3.f1;
        }
        bar(ctr_arg2);
    }
}

// CHECK: p4.action @foo(%arg0: i1) {
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.control @Pipe(%arg0: ui10, %arg1: si16, %arg2: !p4.ref<si16>, %arg3: !p4.ref<si16>)(%arg4: i1, %arg5: si16, %arg6: !p4.header<"MyHeader">) {

// CHECK-NEXT: p4.action @bar(%arg7: si16) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.uninitialized : si16
// CHECK-NEXT: %2 = p4.get_member(%arg6) "f1" : !p4.header<"MyHeader"> -> si16
// CHECK-NEXT: %3 = p4.constant 2 : si64
// CHECK-NEXT: %4 = p4.cast(%3) : si64 -> si16
// CHECK-NEXT: %5 = p4.cmp(%2, %4) ge : (si16, si16) -> i1
// CHECK-NEXT: cf.cond_br %5, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT: %6 = p4.add(%arg5, %arg7) : (si16, si16) -> si16
// CHECK-NEXT: %7 = p4.copy(%6) : si16 -> si16
// CHECK-NEXT: cf.br ^bb3(%7 : si16)
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT: %8 = p4.get_member(%arg6) "f1" : !p4.header<"MyHeader"> -> si16
// CHECK-NEXT: %9 = p4.copy(%8) : si16 -> si16
// CHECK-NEXT: cf.br ^bb3(%9 : si16)
// CHECK-NEXT: ^bb3(%10: si16):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT: p4.store(%arg3, %10) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.call @foo(%arg4) : (i1) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.apply {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.copy(%arg6) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: cf.cond_br %arg4, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT: %2 = p4.get_member(%arg6) "f1" : !p4.header<"MyHeader"> -> si16
// CHECK-NEXT: %3 = p4.add(%arg5, %2) : (si16, si16) -> si16
// CHECK-NEXT: p4.store(%arg2, %3) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: cf.br ^bb2
// CHECK-NEXT: ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT: p4.call @Pipe::@bar(%arg5) : (si16) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: }
