// RUN: p4c-mlir-translate %s | FileCheck %s

enum CounterType {
   Packets,
   Bytes,
   Both
}

enum FooType {
   Packets,
   Bytes,
   Ints
}

extern Counter {
    Counter(bit<32> size, CounterType type);
    void inc(CounterType type);
}

extern CounterType baz(in CounterType ct);

control Pipe(bit<10> arg1, inout CounterType arg2) {
    Counter(32w1024, CounterType.Both) ctr;  // instantiation
    CounterType ctrType;

    action foo() {
        CounterType ct = CounterType.Both;
        if (ct == CounterType.Both) {
            ct = CounterType.Bytes;
        }
        ct = CounterType.Packets;
    }

    action bar(in CounterType arg2, inout CounterType arg3, CounterType arg1) {
        arg3 = baz(arg1);
        arg3 = baz(baz(arg2));
    }

    apply {
        ctr.inc(CounterType.Both);
        bar(CounterType.Both, ctrType, CounterType.Both);
    }
}

// CHECK: module {
// CHECK-NEXT:   p4.enum @CounterType {
// CHECK-NEXT:     p4.enumerator @Packets
// CHECK-NEXT:     p4.enumerator @Bytes
// CHECK-NEXT:     p4.enumerator @Both
// CHECK-NEXT:   }
// CHECK-NEXT:   p4.enum @FooType {
// CHECK-NEXT:     p4.enumerator @Packets
// CHECK-NEXT:     p4.enumerator @Bytes
// CHECK-NEXT:     p4.enumerator @Ints
// CHECK-NEXT:   }
// CHECK-NEXT:   p4.extern_class @Counter {
// CHECK-NEXT:     p4.constructor @Counter_2(ui32, !p4.enum<"CounterType">)
// CHECK-NEXT:     p4.extern @inc_1(!p4.enum<"CounterType">)
// CHECK-NEXT:   }
// CHECK-NEXT:   p4.extern @baz_1(!p4.enum<"CounterType">) -> !p4.enum<"CounterType">
// CHECK-NEXT:   p4.control @Pipe(%arg0: ui10, %arg1: !p4.ref<!p4.enum<"CounterType">>) {
// CHECK-NEXT:     p4.member_decl @ctr : !p4.extern_class<"Counter"> {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:       %1 = p4.constant 1024 : ui32
// CHECK-NEXT:       %2 = p4.constant @CounterType::@Both : !p4.enum<"CounterType">
// CHECK-NEXT:       p4.init @Counter::@Counter_2 !p4.extern_class<"Counter"> (%1, %2) : (ui32, !p4.enum<"CounterType">)
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.action @foo(%arg2: !p4.ref<!p4.enum<"CounterType">>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:       %1 = p4.constant @CounterType::@Both : !p4.enum<"CounterType">
// CHECK-NEXT:       %2 = p4.copy(%1) : !p4.enum<"CounterType"> -> !p4.enum<"CounterType">
// CHECK-NEXT:       %3 = p4.constant @CounterType::@Both : !p4.enum<"CounterType">
// CHECK-NEXT:       %4 = p4.cmp(%2, %3) eq : (!p4.enum<"CounterType">, !p4.enum<"CounterType">) -> i1
// CHECK-NEXT:       cf.cond_br %4, ^bb1, ^bb2(%2 : !p4.enum<"CounterType">)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %5 = p4.constant @CounterType::@Bytes : !p4.enum<"CounterType">
// CHECK-NEXT:       %6 = p4.copy(%5) : !p4.enum<"CounterType"> -> !p4.enum<"CounterType">
// CHECK-NEXT:       cf.br ^bb2(%6 : !p4.enum<"CounterType">)
// CHECK-NEXT:     ^bb2(%7: !p4.enum<"CounterType">):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:       %8 = p4.constant @CounterType::@Packets : !p4.enum<"CounterType">
// CHECK-NEXT:       %9 = p4.copy(%8) : !p4.enum<"CounterType"> -> !p4.enum<"CounterType">
// CHECK-NEXT:       p4.return
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.action @bar(%arg2: !p4.ref<!p4.enum<"CounterType">>, %arg3: !p4.enum<"CounterType">, %arg4: !p4.ref<!p4.enum<"CounterType">>, %arg5: !p4.enum<"CounterType">) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:       %1 = p4.call @baz_1(%arg5) : (!p4.enum<"CounterType">) -> !p4.enum<"CounterType">
// CHECK-NEXT:       p4.store(%arg4, %1) : (!p4.ref<!p4.enum<"CounterType">>, !p4.enum<"CounterType">) -> ()
// CHECK-NEXT:       %2 = p4.call @baz_1(%arg3) : (!p4.enum<"CounterType">) -> !p4.enum<"CounterType">
// CHECK-NEXT:       %3 = p4.call @baz_1(%2) : (!p4.enum<"CounterType">) -> !p4.enum<"CounterType">
// CHECK-NEXT:       p4.store(%arg4, %3) : (!p4.ref<!p4.enum<"CounterType">>, !p4.enum<"CounterType">) -> ()
// CHECK-NEXT:       p4.return
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.apply {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:       %1 = p4.uninitialized : !p4.enum<"CounterType">
// CHECK-NEXT:       %2 = p4.alloc : !p4.ref<!p4.enum<"CounterType">>
// CHECK-NEXT:       p4.store(%2, %1) : (!p4.ref<!p4.enum<"CounterType">>, !p4.enum<"CounterType">) -> ()
// CHECK-NEXT:       %3 = p4.get_member_ref(%0) "ctr" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.extern_class<"Counter">>
// CHECK-NEXT:       %4 = p4.constant @CounterType::@Both : !p4.enum<"CounterType">
// CHECK-NEXT:       p4.call_method %3 @Counter::@inc_1(%4) : (!p4.ref<!p4.extern_class<"Counter">>, !p4.enum<"CounterType">) -> ()
// CHECK-NEXT:       %5 = p4.constant @CounterType::@Both : !p4.enum<"CounterType">
// CHECK-NEXT:       %6 = p4.alloc : !p4.ref<!p4.enum<"CounterType">>
// CHECK-NEXT:       %7 = p4.load(%2) : !p4.ref<!p4.enum<"CounterType">> -> !p4.enum<"CounterType">
// CHECK-NEXT:       p4.store(%6, %7) : (!p4.ref<!p4.enum<"CounterType">>, !p4.enum<"CounterType">) -> ()
// CHECK-NEXT:       %8 = p4.constant @CounterType::@Both : !p4.enum<"CounterType">
// CHECK-NEXT:       p4.call @Pipe::@bar(%2, %5, %6, %8) : (!p4.ref<!p4.enum<"CounterType">>, !p4.enum<"CounterType">, !p4.ref<!p4.enum<"CounterType">>, !p4.enum<"CounterType">) -> ()
// CHECK-NEXT:       %9 = p4.load(%6) : !p4.ref<!p4.enum<"CounterType">> -> !p4.enum<"CounterType">
// CHECK-NEXT:       p4.store(%2, %9) : (!p4.ref<!p4.enum<"CounterType">>, !p4.enum<"CounterType">) -> ()
// CHECK-NEXT:       p4.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
