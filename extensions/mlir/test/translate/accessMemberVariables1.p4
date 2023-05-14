// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
}

control Pipe() {
    int<16> var1;
    MyHeader hdr;

    action foo(inout int<16> arg1) {
        int<16> x1 = var1;
        x1 = 4;
        var1 = x1;
        arg1 = var1 + 1;
        var1 = arg1;
    }

    int<16> var2;

    apply {
        var1 = 3;
        int<16> x1 = 2;
        MyHeader hdr_local;
        if (var1 == 3) {
            int<16> x2 = 3;
            var1 = x1;
        } else {
            x1 = var1;
        }
        var1 = var2 + var1;
        hdr_local = hdr;
        hdr = hdr_local;
        hdr.f1 = 10;
    }
}

// CHECK: p4.header @MyHeader {
// CHECK-NEXT: p4.member_decl @f1 : si16
// CHECK-NEXT: p4.valid_bit_decl @__valid : i1
// CHECK-NEXT: }

// CHECK-NEXT: p4.control @Pipe() {

// CHECK-NEXT: p4.action @foo(%arg0: !p4.ref<si16>, %arg1: !p4.ref<!p4.header<"MyHeader">>, %arg2: !p4.ref<si16>) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.load(%arg0) : !p4.ref<si16> -> si16
// CHECK-NEXT: %2 = p4.copy(%1) : si16 -> si16
// CHECK-NEXT: %3 = p4.constant 4 : si16
// CHECK-NEXT: %4 = p4.cast(%3) : si16 -> si16
// CHECK-NEXT: %5 = p4.copy(%4) : si16 -> si16
// CHECK-NEXT: p4.store(%arg0, %5) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %6 = p4.load(%arg0) : !p4.ref<si16> -> si16
// CHECK-NEXT: %7 = p4.constant 1 : si16
// CHECK-NEXT: %8 = p4.add(%6, %7) : (si16, si16) -> si16
// CHECK-NEXT: p4.store(%arg2, %8) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %9 = p4.load(%arg2) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%arg0, %9) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.apply {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.uninitialized : si16
// CHECK-NEXT: %2 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%2, %1) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %3 = p4.uninitialized : !p4.header<"MyHeader">
// CHECK-NEXT: %4 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%4, %3) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %5 = p4.uninitialized : si16
// CHECK-NEXT: %6 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%6, %5) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %7 = p4.constant 3 : si16
// CHECK-NEXT: %8 = p4.cast(%7) : si16 -> si16
// CHECK-NEXT: p4.store(%2, %8) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %9 = p4.constant 2 : si16
// CHECK-NEXT: %10 = p4.cast(%9) : si16 -> si16
// CHECK-NEXT: %11 = p4.copy(%10) : si16 -> si16
// CHECK-NEXT: %12 = p4.uninitialized : !p4.header<"MyHeader">
// CHECK-NEXT: %13 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%13, %12) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %14 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: %15 = p4.constant 3 : si64
// CHECK-NEXT: %16 = p4.cast(%15) : si64 -> si16
// CHECK-NEXT: %17 = p4.cmp(%14, %16) eq : (si16, si16) -> i1
// CHECK-NEXT: cf.cond_br %17, ^bb1, ^bb2

    // CHECK-NEXT: ^bb1:  // pred: ^bb0
    // CHECK-NEXT: %18 = p4.constant 3 : si16
    // CHECK-NEXT: %19 = p4.cast(%18) : si16 -> si16
    // CHECK-NEXT: %20 = p4.copy(%19) : si16 -> si16
    // CHECK-NEXT: p4.store(%2, %11) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: cf.br ^bb3(%11 : si16)

    // CHECK-NEXT: ^bb2:  // pred: ^bb0
    // CHECK-NEXT: %21 = p4.load(%2) : !p4.ref<si16> -> si16
    // CHECK-NEXT: %22 = p4.copy(%21) : si16 -> si16
    // CHECK-NEXT: cf.br ^bb3(%22 : si16)

// CHECK-NEXT: ^bb3(%23: si16):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT: %24 = p4.load(%6) : !p4.ref<si16> -> si16
// CHECK-NEXT: %25 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: %26 = p4.add(%24, %25) : (si16, si16) -> si16
// CHECK-NEXT: p4.store(%2, %26) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %27 = p4.load(%4) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: p4.store(%13, %27) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %28 = p4.load(%13) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: p4.store(%4, %28) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %29 = p4.get_member_ref(%4) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: %30 = p4.constant 10 : si16
// CHECK-NEXT: %31 = p4.cast(%30) : si16 -> si16
// CHECK-NEXT: p4.store(%29, %31) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }




