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

    action bar(int<16> x1, int<16> x2, int<16> x3) {}

    int<16> test;

    table myTable {
        key = {
            arg1.f1 : exact;
            arg3 : lpm;
            cst : lpm;
            // TODO: test : exact;
        }
        // TODO: prop4 = Ext(test);
        // TODO: prop5 = baz(test);
        actions = { foo(arg2, arg3); bar; bak; }
        prop1 = 42;
        prop2 = baz(cst);
        prop3 = baz(ctr_arg1 + 3);
    }

    apply {
        // TODO: myTable.apply()
    }
}
