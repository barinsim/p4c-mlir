// RUN: p4c-mlir-translate %s | FileCheck %s

extern bool foo();

control TopPipe(in bool flag1, bool flag2)(bool flag3) {
    const bool f1 = flag2 || flag3;
    const bool f2 = flag2 && flag3;
    const bool f3 = flag2 && flag3 || flag3;
    const bool f4 = foo() || foo();
    const bool f7 = flag2 || flag3 || flag3;
    const bool f8 = flag2 || flag3 && flag3;

    apply {
        bool f5 = flag2 || flag3;
        if (f5 == false) {
            f5 = true;
        } else {
            f5 = false && f1;
        }
        f5 = f1 && f5 || flag1;
        bool f6 = flag1 || flag2;
    }
}

// CHECK: module {
// CHECK-NEXT:   p4.extern @foo_0() -> i1
// CHECK-NEXT:   p4.control @TopPipe(%arg0: i1, %arg1: i1)(%arg2: i1) {
// CHECK-NEXT:     p4.member_decl @f1 : i1 {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"TopPipe">>
// CHECK-NEXT:       %1 = p4.constant true : i1
// CHECK-NEXT:       %2 = p4.cmp(%arg1, %1) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %2, ^bb1, ^bb2(%1 : i1)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       cf.br ^bb2(%arg2 : i1)
// CHECK-NEXT:     ^bb2(%3: i1):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:       %4 = p4.copy(%3) : i1 -> i1
// CHECK-NEXT:       p4.init i1 (%4) : (i1)
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.member_decl @f2 : i1 {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"TopPipe">>
// CHECK-NEXT:       %1 = p4.constant false : i1
// CHECK-NEXT:       %2 = p4.cmp(%arg1, %1) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %2, ^bb1, ^bb2(%1 : i1)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       cf.br ^bb2(%arg2 : i1)
// CHECK-NEXT:     ^bb2(%3: i1):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:       %4 = p4.copy(%3) : i1 -> i1
// CHECK-NEXT:       p4.init i1 (%4) : (i1)
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.member_decl @f3 : i1 {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"TopPipe">>
// CHECK-NEXT:       %1 = p4.constant false : i1
// CHECK-NEXT:       %2 = p4.cmp(%arg1, %1) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %2, ^bb1, ^bb2(%1 : i1)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       cf.br ^bb2(%arg2 : i1)
// CHECK-NEXT:     ^bb2(%3: i1):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:       %4 = p4.copy(%3) : i1 -> i1
// CHECK-NEXT:       %5 = p4.constant true : i1
// CHECK-NEXT:       %6 = p4.cmp(%4, %5) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %6, ^bb3, ^bb4(%5 : i1)
// CHECK-NEXT:     ^bb3:  // pred: ^bb2
// CHECK-NEXT:       cf.br ^bb4(%arg2 : i1)
// CHECK-NEXT:     ^bb4(%7: i1):  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:       %8 = p4.copy(%7) : i1 -> i1
// CHECK-NEXT:       p4.init i1 (%8) : (i1)
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.member_decl @f4 : i1 {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"TopPipe">>
// CHECK-NEXT:       %1 = p4.call @foo_0() : () -> i1
// CHECK-NEXT:       %2 = p4.constant true : i1
// CHECK-NEXT:       %3 = p4.cmp(%1, %2) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %3, ^bb1, ^bb2(%2 : i1)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %4 = p4.call @foo_0() : () -> i1
// CHECK-NEXT:       cf.br ^bb2(%4 : i1)
// CHECK-NEXT:     ^bb2(%5: i1):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:       %6 = p4.copy(%5) : i1 -> i1
// CHECK-NEXT:       p4.init i1 (%6) : (i1)
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.member_decl @f7 : i1 {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"TopPipe">>
// CHECK-NEXT:       %1 = p4.constant true : i1
// CHECK-NEXT:       %2 = p4.cmp(%arg1, %1) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %2, ^bb1, ^bb2(%1 : i1)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       cf.br ^bb2(%arg2 : i1)
// CHECK-NEXT:     ^bb2(%3: i1):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:       %4 = p4.copy(%3) : i1 -> i1
// CHECK-NEXT:       %5 = p4.constant true : i1
// CHECK-NEXT:       %6 = p4.cmp(%4, %5) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %6, ^bb3, ^bb4(%5 : i1)
// CHECK-NEXT:     ^bb3:  // pred: ^bb2
// CHECK-NEXT:       cf.br ^bb4(%arg2 : i1)
// CHECK-NEXT:     ^bb4(%7: i1):  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:       %8 = p4.copy(%7) : i1 -> i1
// CHECK-NEXT:       p4.init i1 (%8) : (i1)
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.member_decl @f8 : i1 {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"TopPipe">>
// CHECK-NEXT:       %1 = p4.constant true : i1
// CHECK-NEXT:       %2 = p4.cmp(%arg1, %1) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %2, ^bb1, ^bb4(%1 : i1)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %3 = p4.constant false : i1
// CHECK-NEXT:       %4 = p4.cmp(%arg2, %3) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %4, ^bb2, ^bb3(%3 : i1)
// CHECK-NEXT:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       cf.br ^bb3(%arg2 : i1)
// CHECK-NEXT:     ^bb3(%5: i1):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:       %6 = p4.copy(%5) : i1 -> i1
// CHECK-NEXT:       cf.br ^bb4(%6 : i1)
// CHECK-NEXT:     ^bb4(%7: i1):  // 2 preds: ^bb0, ^bb3
// CHECK-NEXT:       %8 = p4.copy(%7) : i1 -> i1
// CHECK-NEXT:       p4.init i1 (%8) : (i1)
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.apply {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.control<"TopPipe">>
// CHECK-NEXT:       %1 = p4.constant true : i1
// CHECK-NEXT:       %2 = p4.cmp(%arg1, %1) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %2, ^bb1, ^bb2(%1 : i1)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       cf.br ^bb2(%arg2 : i1)
// CHECK-NEXT:     ^bb2(%3: i1):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:       %4 = p4.copy(%3) : i1 -> i1
// CHECK-NEXT:       %5 = p4.copy(%4) : i1 -> i1
// CHECK-NEXT:       %6 = p4.constant false : i1
// CHECK-NEXT:       %7 = p4.cmp(%5, %6) eq : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %7, ^bb3, ^bb4
// CHECK-NEXT:     ^bb3:  // pred: ^bb2
// CHECK-NEXT:       %8 = p4.constant true : i1
// CHECK-NEXT:       %9 = p4.copy(%8) : i1 -> i1
// CHECK-NEXT:       cf.br ^bb7(%9 : i1)
// CHECK-NEXT:     ^bb4:  // pred: ^bb2
// CHECK-NEXT:       %10 = p4.constant false : i1
// CHECK-NEXT:       %11 = p4.constant false : i1
// CHECK-NEXT:       %12 = p4.cmp(%10, %11) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %12, ^bb5, ^bb6(%11 : i1)
// CHECK-NEXT:     ^bb5:  // pred: ^bb4
// CHECK-NEXT:       %13 = p4.get_member_ref(%0) "f1" : !p4.ref<!p4.control<"TopPipe">> -> !p4.ref<i1>
// CHECK-NEXT:       %14 = p4.load(%13) : !p4.ref<i1> -> i1
// CHECK-NEXT:       cf.br ^bb6(%14 : i1)
// CHECK-NEXT:     ^bb6(%15: i1):  // 2 preds: ^bb4, ^bb5
// CHECK-NEXT:       %16 = p4.copy(%15) : i1 -> i1
// CHECK-NEXT:       %17 = p4.copy(%16) : i1 -> i1
// CHECK-NEXT:       cf.br ^bb7(%17 : i1)
// CHECK-NEXT:     ^bb7(%18: i1):  // 2 preds: ^bb3, ^bb6
// CHECK-NEXT:       %19 = p4.get_member_ref(%0) "f1" : !p4.ref<!p4.control<"TopPipe">> -> !p4.ref<i1>
// CHECK-NEXT:       %20 = p4.load(%19) : !p4.ref<i1> -> i1
// CHECK-NEXT:       %21 = p4.constant false : i1
// CHECK-NEXT:       %22 = p4.cmp(%20, %21) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %22, ^bb8, ^bb9(%21 : i1)
// CHECK-NEXT:     ^bb8:  // pred: ^bb7
// CHECK-NEXT:       cf.br ^bb9(%18 : i1)
// CHECK-NEXT:     ^bb9(%23: i1):  // 2 preds: ^bb7, ^bb8
// CHECK-NEXT:       %24 = p4.copy(%23) : i1 -> i1
// CHECK-NEXT:       %25 = p4.constant true : i1
// CHECK-NEXT:       %26 = p4.cmp(%24, %25) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %26, ^bb10, ^bb11(%25 : i1)
// CHECK-NEXT:     ^bb10:  // pred: ^bb9
// CHECK-NEXT:       cf.br ^bb11(%arg0 : i1)
// CHECK-NEXT:     ^bb11(%27: i1):  // 2 preds: ^bb9, ^bb10
// CHECK-NEXT:       %28 = p4.copy(%27) : i1 -> i1
// CHECK-NEXT:       %29 = p4.copy(%28) : i1 -> i1
// CHECK-NEXT:       %30 = p4.constant true : i1
// CHECK-NEXT:       %31 = p4.cmp(%arg0, %30) ne : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %31, ^bb12, ^bb13(%30 : i1)
// CHECK-NEXT:     ^bb12:  // pred: ^bb11
// CHECK-NEXT:       cf.br ^bb13(%arg1 : i1)
// CHECK-NEXT:     ^bb13(%32: i1):  // 2 preds: ^bb11, ^bb12
// CHECK-NEXT:       %33 = p4.copy(%32) : i1 -> i1
// CHECK-NEXT:       %34 = p4.copy(%33) : i1 -> i1
// CHECK-NEXT:       p4.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
