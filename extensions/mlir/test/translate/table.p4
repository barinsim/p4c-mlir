// RUN: p4c-mlir-translate %s | FileCheck %s

match_kind {
    exact,
    lpm
}

header MyHeader {
    int<16> f1;
}

extern Ext {
    Ext(int<16> tmp);
}

extern int<16> baz(in int<16> arg);

action bak() {}

control Pipe(in MyHeader arg1, in int<16> arg2, inout int<16> arg3)(int<16> ctr_arg1) {

    action foo(in int<16> x1, inout int<16> x2) {}

    const int<16> cst = 3;

    MyHeader local_hdr = {23};

    action bar(int<16> x1, int<16> x2, int<16> x3) {
        local_hdr.f1 = arg2;
    }

    int<16> test = 52;

    table myTable {
        key = {
            arg1.f1 : exact;
            arg3 : lpm;
            cst : lpm;
            test : exact;
            local_hdr.f1 : lpm;
        }
        actions = { foo(arg2, arg3); bar; bak(); }
        const default_action = bak();
        prop1 = 42;
        prop2 = baz(cst);
        prop3 = baz(ctr_arg1 + 3);
        prop5 = baz(test);
    }

    apply {
        myTable.apply();
        test = test + 1;
        local_hdr.f1 = 3;
    }
}

// CHECK: module {
// CHECK-NEXT: p4.match_kind @exact
// CHECK-NEXT: p4.match_kind @lpm

// CHECK-NEXT: p4.header @MyHeader {
// CHECK-NEXT: p4.member_decl @f1 : si16
// CHECK-NEXT: p4.valid_bit_decl @__valid : i1
// CHECK-NEXT: }

// CHECK-NEXT: p4.extern_class @Ext {
// CHECK-NEXT: p4.constructor @Ext_1(si16)
// CHECK-NEXT: }

// CHECK-NEXT: p4.extern @baz_1(si16) -> si16

// CHECK-NEXT: p4.action @bak() {
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.control @Pipe(%arg0: !p4.header<"MyHeader">, %arg1: si16, %arg2: !p4.ref<si16>)(%arg3: si16) {

// CHECK-NEXT: p4.action @foo(%arg4: si16, %arg5: !p4.ref<si16>) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @cst : si16 {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant 3 : si16
// CHECK-NEXT: %2 = p4.cast(%1) : si16 -> si16
// CHECK-NEXT: p4.init si16 (%2) : (si16)
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @bar(%arg4: !p4.ref<!p4.header<"MyHeader">>, %arg5: si16, %arg6: si16, %arg7: si16) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.get_member_ref(%arg4) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%1, %arg1) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.table @myTable(%arg4: !p4.ref<!p4.header<"MyHeader">>, %arg5: !p4.ref<si16>) {
    // CHECK-NEXT: p4.table_keys_list {

    // CHECK-NEXT: p4.table_key match @exact {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.get_member(%arg0) "f1" : !p4.header<"MyHeader"> -> si16
    // CHECK-NEXT: p4.init si16 (%1) : (si16)
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_key match @lpm {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.load(%arg2) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.init si16 (%1) : (si16)
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_key match @lpm {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.get_member_ref(%0) "cst" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
    // CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.init si16 (%2) : (si16)
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_key match @exact {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.load(%arg5) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.init si16 (%1) : (si16)
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_key match @lpm {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.get_member_ref(%arg4) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.init si16 (%2) : (si16)
    // CHECK-NEXT: }

    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_actions_list {

    // CHECK-NEXT: p4.table_action {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.alloc : !p4.ref<si16>
    // CHECK-NEXT: %2 = p4.load(%arg2) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.store(%1, %2) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: p4.call @Pipe::@foo(%arg1, %1) : (si16, !p4.ref<si16>) -> ()
    // CHECK-NEXT: %3 = p4.load(%1) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.store(%arg2, %3) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_action @Pipe::@bar

    // CHECK-NEXT: p4.table_action {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: p4.call @bak() : () -> ()
    // CHECK-NEXT: }

    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_default_action {
        // CHECK-NEXT: p4.table_action {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: p4.call @bak() : () -> ()
        // CHECK-NEXT: }
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_property @prop1 {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.constant 42 : si64
    // CHECK-NEXT: p4.init si64 (%1) : (si64)
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_property @prop2 {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.get_member_ref(%0) "cst" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
    // CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<si16> -> si16
    // CHECK-NEXT: %3 = p4.call @baz_1(%2) : (si16) -> si16
    // CHECK-NEXT: p4.init si16 (%3) : (si16)
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_property @prop3 {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.constant 3 : si16
    // CHECK-NEXT: %2 = p4.add(%arg3, %1) : (si16, si16) -> si16
    // CHECK-NEXT: %3 = p4.call @baz_1(%2) : (si16) -> si16
    // CHECK-NEXT: p4.init si16 (%3) : (si16)
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_property @prop5 {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.load(%arg5) : !p4.ref<si16> -> si16
    // CHECK-NEXT: %2 = p4.call @baz_1(%1) : (si16) -> si16
    // CHECK-NEXT: p4.init si16 (%2) : (si16)
    // CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK-NEXT: p4.apply {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant 23 : si16
// CHECK-NEXT: %2 = p4.constant true
// CHECK-NEXT: %3 = p4.tuple(%1, %2) : (si16, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %4 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%4, %3) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %5 = p4.constant 52 : si16
// CHECK-NEXT: %6 = p4.cast(%5) : si16 -> si16
// CHECK-NEXT: %7 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%7, %6) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %8 = p4.get_member_ref(%0) "myTable" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.table<"myTable">>
// CHECK-NEXT: p4.call_apply %8 (%4, %7) : (!p4.ref<!p4.table<"myTable">>, !p4.ref<!p4.header<"MyHeader">>, !p4.ref<si16>) -> ()
// CHECK-NEXT: %9 = p4.load(%7) : !p4.ref<si16> -> si16
// CHECK-NEXT: %10 = p4.constant 1 : si16
// CHECK-NEXT: %11 = p4.add(%9, %10) : (si16, si16) -> si16
// CHECK-NEXT: p4.store(%7, %11) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %12 = p4.get_member_ref(%4) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: %13 = p4.constant 3 : si16
// CHECK-NEXT: %14 = p4.cast(%13) : si16 -> si16
// CHECK-NEXT: p4.store(%12, %14) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: }
// CHECK-NEXT: }



