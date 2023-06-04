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

action bak() {}

control Pipe(in MyHeader arg1, in int<16> arg2, inout int<16> arg3)(int<16> ctr_arg1) {

    action foo(in int<16> x1, in int<16> x2) {}

    const int<16> cst = 3;

    MyHeader local_hdr = {23};

    action bar(int<16> x1, int<16> x2, int<16> x3) {
        local_hdr.f1 = arg2;
    }

    int<16> test = 52;

    table myTable {
        key = {
            test : exact;
            cst : lpm;
        }
        actions = {
            bar(); bak(); foo(cst, 3);
        }
        const entries = {
            (0x01, 0x1111) : foo(cst, 3);
            (0x02, 0x1181 + cst) : bar(4, 2, 1);
            (ctr_arg1, ctr_arg1) : bak();
            (0x04, _) : bak();
            // TODO: (0x03, 0x1111 &&& 0x02F0) : bak();
        }
        // TODO: prop = Ext(test);
    }

    apply {
        myTable.apply();
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

// CHECK-NEXT: p4.action @bak() {
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.control @Pipe(%arg0: !p4.header<"MyHeader">, %arg1: si16, %arg2: !p4.ref<si16>)(%arg3: si16) {

// CHECK-NEXT: p4.action @foo(%arg4: si16, %arg5: si16) {
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
    // CHECK-NEXT: %1 = p4.load(%arg5) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.init si16 (%1) : (si16)
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_key match @lpm {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.get_member_ref(%0) "cst" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
    // CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.init si16 (%2) : (si16)
    // CHECK-NEXT: }

// CHECK-NEXT: }

// CHECK-NEXT: p4.table_actions_list {

    // CHECK-NEXT: p4.table_action {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.control_plane_value : si16
    // CHECK-NEXT: %2 = p4.control_plane_value : si16
    // CHECK-NEXT: %3 = p4.control_plane_value : si16
    // CHECK-NEXT: p4.call @Pipe::@bar(%arg4, %1, %2, %3) : (!p4.ref<!p4.header<"MyHeader">>, si16, si16, si16) -> ()
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_action {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: p4.call @bak() : () -> ()
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_action {
    // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
    // CHECK-NEXT: %1 = p4.get_member_ref(%0) "cst" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
    // CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<si16> -> si16
    // CHECK-NEXT: %3 = p4.constant 3 : si16
    // CHECK-NEXT: %4 = p4.cast(%3) : si16 -> si16
    // CHECK-NEXT: p4.call @Pipe::@foo(%2, %4) : (si16, si16) -> ()
    // CHECK-NEXT: }

// CHECK-NEXT: }

// CHECK-NEXT: p4.table_entries_list {
    // CHECK-NEXT: p4.table_entry {
        // CHECK-NEXT: p4.table_entry_keys {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: %1 = p4.constant 1 : si16
        // CHECK-NEXT: %2 = p4.constant 4369 : si16
        // CHECK-NEXT: %3 = p4.tuple(%1, %2) : (si16, si16) -> tuple<si16, si16>
        // CHECK-NEXT: p4.init tuple<si16, si16> (%3) : (tuple<si16, si16>)
        // CHECK-NEXT: }
        // CHECK-NEXT: p4.table_action {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: %1 = p4.get_member_ref(%0) "cst" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<si16> -> si16
        // CHECK-NEXT: %3 = p4.constant 3 : si16
        // CHECK-NEXT: %4 = p4.cast(%3) : si16 -> si16
        // CHECK-NEXT: p4.call @Pipe::@foo(%2, %4) : (si16, si16) -> ()
        // CHECK-NEXT: }
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_entry {
        // CHECK-NEXT: p4.table_entry_keys {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: %1 = p4.constant 2 : si16
        // CHECK-NEXT: %2 = p4.constant 4481 : si16
        // CHECK-NEXT: %3 = p4.get_member_ref(%0) "cst" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
        // CHECK-NEXT: %4 = p4.load(%3) : !p4.ref<si16> -> si16
        // CHECK-NEXT: %5 = p4.add(%2, %4) : (si16, si16) -> si16
        // CHECK-NEXT: %6 = p4.tuple(%1, %5) : (si16, si16) -> tuple<si16, si16>
        // CHECK-NEXT: p4.init tuple<si16, si16> (%6) : (tuple<si16, si16>)
        // CHECK-NEXT: }
        // CHECK-NEXT: p4.table_action {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: %1 = p4.constant 4 : si16
        // CHECK-NEXT: %2 = p4.constant 2 : si16
        // CHECK-NEXT: %3 = p4.constant 1 : si16
        // CHECK-NEXT: p4.call @Pipe::@bar(%arg4, %1, %2, %3) : (!p4.ref<!p4.header<"MyHeader">>, si16, si16, si16) -> ()
        // CHECK-NEXT: }
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_entry {
        // CHECK-NEXT: p4.table_entry_keys {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: %1 = p4.tuple(%arg3, %arg3) : (si16, si16) -> tuple<si16, si16>
        // CHECK-NEXT: p4.init tuple<si16, si16> (%1) : (tuple<si16, si16>)
        // CHECK-NEXT: }
        // CHECK-NEXT: p4.table_action {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: p4.call @bak() : () -> ()
        // CHECK-NEXT: }
    // CHECK-NEXT: }

    // CHECK-NEXT: p4.table_entry {
        // CHECK-NEXT: p4.table_entry_keys {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: %1 = p4.constant 4 : si16
        // CHECK-NEXT: %2 = p4.dontcare : !p4.dontcare
        // CHECK-NEXT: %3 = p4.tuple(%1, %2) : (si16, !p4.dontcare) -> tuple<si16, !p4.dontcare>
        // CHECK-NEXT: p4.init tuple<si16, !p4.dontcare> (%3) : (tuple<si16, !p4.dontcare>)
        // CHECK-NEXT: }
        // CHECK-NEXT: p4.table_action {
        // CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
        // CHECK-NEXT: p4.call @bak() : () -> ()
    // CHECK-NEXT: }
// CHECK-NEXT: }
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
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: }
// CHECK-NEXT: }
