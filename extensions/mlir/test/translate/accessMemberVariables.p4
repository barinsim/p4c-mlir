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
    }
}

// CHECK: p4.control @Pipe() {

    // CHECK-NEXT: p4.member_decl @var1 : si16
    // CHECK-NEXT: p4.member_decl @hdr : !p4.header<"MyHeader">

    // CHECK-NEXT: p4.action @foo(%arg0: !p4.ref<si16>) {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: %1 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<si16> -> si16
        // CHECK-NEXT: %3 = p4.copy(%2) : si16 -> si16
        // CHECK-NEXT: %4 = p4.constant 4 : si16
        // CHECK-NEXT: %5 = p4.cast(%4) : si16 -> si16
        // CHECK-NEXT: %6 = p4.copy(%5) : si16 -> si16
        // CHECK-NEXT: %7 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: p4.store(%7, %6) : (!p4.ref<si16>, si16) -> ()
        // CHECK-NEXT: %8 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %9 = p4.load(%8) : !p4.ref<si16> -> si16
        // CHECK-NEXT: %10 = p4.constant 1 : si16
        // CHECK-NEXT: %11 = p4.add(%9, %10) : (si16, si16) -> si16
        // CHECK-NEXT: p4.store(%arg0, %11) : (!p4.ref<si16>, si16) -> ()
        // CHECK-NEXT: %12 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %13 = p4.load(%arg0) : !p4.ref<si16> -> si16
        // CHECK-NEXT: p4.store(%12, %13) : (!p4.ref<si16>, si16) -> ()
        // CHECK-NEXT: p4.return
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.member_decl @var2 : si16

    // CHECK-NEXT: p4.apply {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: %1 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %2 = p4.constant 3 : si16
        // CHECK-NEXT: %3 = p4.cast(%2) : si16 -> si16
        // CHECK-NEXT: p4.store(%1, %3) : (!p4.ref<si16>, si16) -> ()
        // CHECK-NEXT: %4 = p4.constant 2 : si16
        // CHECK-NEXT: %5 = p4.cast(%4) : si16 -> si16
        // CHECK-NEXT: %6 = p4.copy(%5) : si16 -> si16
        // CHECK-NEXT: %7 = p4.uninitialized : !p4.header<"MyHeader">
        // CHECK-NEXT: %8 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %9 = p4.load(%8) : !p4.ref<si16> -> si16
        // CHECK-NEXT: %10 = p4.constant 3 : si64
        // CHECK-NEXT: %11 = p4.cast(%10) : si64 -> si16
        // CHECK-NEXT: %12 = p4.cmp(%9, %11) eq : (si16, si16) -> i1
        // CHECK-NEXT: cf.cond_br %12, ^bb1, ^bb2

            // CHECK: ^bb1:
            // CHECK-NEXT: %13 = p4.constant 3 : si16
            // CHECK-NEXT: %14 = p4.cast(%13) : si16 -> si16
            // CHECK-NEXT: %15 = p4.copy(%14) : si16 -> si16
            // CHECK-NEXT: %16 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
            // CHECK-NEXT: p4.store(%16, %6) : (!p4.ref<si16>, si16) -> ()
            // CHECK-NEXT: cf.br ^bb3(%6 : si16)

            // CHECK: ^bb2:
            // CHECK-NEXT: %17 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
            // CHECK-NEXT: %18 = p4.load(%17) : !p4.ref<si16> -> si16
            // CHECK-NEXT: %19 = p4.copy(%18) : si16 -> si16
            // CHECK-NEXT: cf.br ^bb3(%19 : si16)

        // CHECK-NEXT: ^bb3(%20: si16):
        // CHECK-NEXT: %21 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %22 = p4.get_member_ref(%0) "var2" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %23 = p4.load(%22) : !p4.ref<si16> -> si16
        // CHECK-NEXT: %24 = p4.get_member_ref(%0) "var1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %25 = p4.load(%24) : !p4.ref<si16> -> si16
        // CHECK-NEXT: %26 = p4.add(%23, %25) : (si16, si16) -> si16
        // CHECK-NEXT: p4.store(%21, %26) : (!p4.ref<si16>, si16) -> ()
        // CHECK-NEXT: %27 = p4.get_member_ref(%0) "hdr" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
        // CHECK-NEXT: %28 = p4.load(%27) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
        // CHECK-NEXT: %29 = p4.copy(%28) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
        // CHECK-NEXT: %30 = p4.get_member_ref(%0) "hdr" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
        // CHECK-NEXT: p4.store(%30, %29) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
        // CHECK-NEXT: p4.return
    // CHECK-NEXT: }

// CHECK-NEXT: }


