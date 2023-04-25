// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
    bit<16> f2;
    bit<10> f3;
}

// CHECK: p4.header @MyHeader {
// CHECK-NEXT: p4.member_decl @f1 : si16
// CHECK-NEXT: p4.member_decl @f2 : ui16
// CHECK-NEXT: p4.member_decl @f3 : ui10
// CHECK-NEXT: p4.valid_bit_decl @__valid : i1
// CHECK-NEXT: }

struct MyInnerInnerStruct {
    int<10> f1;
    MyHeader f2;
    MyHeader f3;
}

// CHECK-NEXT: p4.struct @MyInnerInnerStruct {
// CHECK-NEXT: p4.member_decl @f1 : si10
// CHECK-NEXT: p4.member_decl @f2 : !p4.header<"MyHeader">
// CHECK-NEXT: p4.member_decl @f3 : !p4.header<"MyHeader">
// CHECK-NEXT: }

struct MyInnerStruct {
    int<10> f1;
    MyHeader f2;
    MyInnerInnerStruct f3;
}

// CHECK-NEXT: p4.struct @MyInnerStruct {
// CHECK-NEXT: p4.member_decl @f1 : si10
// CHECK-NEXT: p4.member_decl @f2 : !p4.header<"MyHeader">
// CHECK-NEXT: p4.member_decl @f3 : !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: }

struct MyStruct {
    MyInnerStruct f1;
    MyInnerStruct f2;
    bit<10> f3;
}

// CHECK-NEXT: p4.struct @MyStruct {
// CHECK-NEXT: p4.member_decl @f1 : !p4.struct<"MyInnerStruct">
// CHECK-NEXT: p4.member_decl @f2 : !p4.struct<"MyInnerStruct">
// CHECK-NEXT: p4.member_decl @f3 : ui10
// CHECK-NEXT: }

struct EmptyStruct {}

// CHECK-NEXT: p4.struct @EmptyStruct {
// CHECK-NEXT: }

action foo() {
    MyStruct str;
}

// CHECK-NEXT: p4.action @foo() {
// CHECK-NEXT: %0 = p4.uninitialized : !p4.struct<"MyStruct">
// CHECK-NEXT: p4.return
// CHECK-NEXT: }
