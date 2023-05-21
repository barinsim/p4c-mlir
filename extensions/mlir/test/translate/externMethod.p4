// RUN: p4c-mlir-translate %s | FileCheck %s

extern packet_out {
    void emit<T>(in T data);
}

extern packet_in {
    packet_in();
    void extract<T>(out T hdr);
    void extract<T>(out T variableSizeHeader, in bit<32> variableFieldSizeInBits);
    T lookahead<T>();
    void advance(inout bit<32> sizeInBits);
    bit<32> length();
}

extern Checksum<T> {
    Checksum();
    Checksum(T arg);
    void clear();
    void update<T>(in T data);
    bool remove<T>(in T data);
    bit<16> get();
}

extern Register<T, R> {
    Register(T size, R tmp);
    Register(R arg, T size, T tmp);
    T read(bit<32> index);
    void write(bit<32> index, in int<32> value);
    T lookahead<T>();
    R lookahead<T>(R flag);
}

header MyHeader {
    int<32> f1;
}

control Pipe() {

    packet_in() pkt;

    Register((int<32>)10, true, false) reg1;
    Register<int<32>, bool>(10 + 2, false) reg2;
    Register<bool, MyHeader>({42 + 3 + 1}, true, false) reg3;

    Checksum<Register<int<32>, bool>>() chk1;
    // TODO: Checksum(reg1) chk2;

    const MyHeader hdr = {50};
    Checksum(hdr) chk3;
    Checksum(hdr.f1) chk4;

    apply {

    }
}

// TODO: call methods, pass externs as type params and arguments, constructor

// CHECK: module {

// CHECK-NEXT: p4.extern_class @packet_out {
// CHECK-NEXT: p4.extern @emit_1<@T>(!p4.type_var<@T>)
// CHECK-NEXT: }

// CHECK-NEXT: p4.extern_class @packet_in {
// CHECK-NEXT: p4.extern @packet_in_0()
// CHECK-NEXT: p4.extern @extract_1<@T>(!p4.ref<!p4.type_var<@T>>)
// CHECK-NEXT: p4.extern @extract_2<@T>(!p4.ref<!p4.type_var<@T>>, ui32)
// CHECK-NEXT: p4.extern @lookahead_0<@T>() -> !p4.type_var<@T>
// CHECK-NEXT: p4.extern @advance_1(!p4.ref<ui32>)
// CHECK-NEXT: p4.extern @length_0() -> ui32
// CHECK-NEXT: }

// CHECK-NEXT: p4.extern_class @Checksum<@T> {
// CHECK-NEXT: p4.extern @Checksum_0()
// CHECK-NEXT: p4.extern @Checksum_1(!p4.type_var<@T>)
// CHECK-NEXT: p4.extern @clear_0()
// CHECK-NEXT: p4.extern @update_1<@T>(!p4.type_var<@T>)
// CHECK-NEXT: p4.extern @remove_1<@T>(!p4.type_var<@T>) -> i1
// CHECK-NEXT: p4.extern @get_0() -> ui16
// CHECK-NEXT: }

// CHECK-NEXT: p4.extern_class @Register<@T, @R> {
// CHECK-NEXT: p4.extern @Register_2(!p4.type_var<@T>, !p4.type_var<@R>)
// CHECK-NEXT: p4.extern @Register_3(!p4.type_var<@R>, !p4.type_var<@T>, !p4.type_var<@T>)
// CHECK-NEXT: p4.extern @read_1(ui32) -> !p4.type_var<@T>
// CHECK-NEXT: p4.extern @write_2(ui32, si32)
// CHECK-NEXT: p4.extern @lookahead_0<@T>() -> !p4.type_var<@T>
// CHECK-NEXT: p4.extern @lookahead_1<@T>(!p4.type_var<@R>) -> !p4.type_var<@R>
// CHECK-NEXT: }

// CHECK-NEXT: p4.header @MyHeader {
// CHECK-NEXT: p4.member_decl @f1 : si32
// CHECK-NEXT: p4.valid_bit_decl @__valid : i1
// CHECK-NEXT: }

// CHECK-NEXT: p4.control @Pipe() {
// CHECK-NEXT: p4.member_decl @pkt : !p4.extern_class<"packet_in"> {
// CHECK-NEXT:   %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:   p4.init !p4.extern_class<"packet_in"> () : ()
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @reg1 : !p4.extern_class<"Register"<i1, si32>> {
// CHECK-NEXT:   %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:   %1 = p4.constant 10 : si64
// CHECK-NEXT:   %2 = p4.cast(%1) : si64 -> si32
// CHECK-NEXT:   %3 = p4.constant true
// CHECK-NEXT:   %4 = p4.constant false
// CHECK-NEXT:   p4.init !p4.extern_class<"Register"<i1, si32>> (%2, %3, %4) : (si32, i1, i1)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @reg2 : !p4.extern_class<"Register"<si32, i1>> {
// CHECK-NEXT:   %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:   %1 = p4.constant 12 : si64
// CHECK-NEXT:   %2 = p4.cast(%1) : si64 -> si32
// CHECK-NEXT:   %3 = p4.constant false
// CHECK-NEXT:   p4.init !p4.extern_class<"Register"<si32, i1>> (%2, %3) : (si32, i1)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @reg3 : !p4.extern_class<"Register"<i1, !p4.header<"MyHeader">>> {
// CHECK-NEXT:   %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:   %1 = p4.constant 46 : si64
// CHECK-NEXT:   %2 = p4.cast(%1) : si64 -> si32
// CHECK-NEXT:   %3 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT:   %4 = p4.get_member_ref(%3) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si32>
// CHECK-NEXT:   p4.store(%4, %2) : (!p4.ref<si32>, si32) -> ()
// CHECK-NEXT:   %5 = p4.get_member_ref(%3) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT:   %6 = p4.constant true
// CHECK-NEXT:   p4.store(%5, %6) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT:   %7 = p4.load(%3) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT:   %8 = p4.constant true
// CHECK-NEXT:   %9 = p4.constant false
// CHECK-NEXT:   p4.init !p4.extern_class<"Register"<i1, !p4.header<"MyHeader">>> (%7, %8, %9) : (!p4.header<"MyHeader">, i1, i1)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @chk1 : !p4.extern_class<"Checksum"<!p4.extern_class<"Register"<si32, i1>>>> {
// CHECK-NEXT:   %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:   p4.init !p4.extern_class<"Checksum"<!p4.extern_class<"Register"<si32, i1>>>> () : ()
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @hdr : !p4.header<"MyHeader"> {
// CHECK-NEXT:   %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:   %1 = p4.constant 50 : si32
// CHECK-NEXT:   %2 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT:   %3 = p4.get_member_ref(%2) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si32>
// CHECK-NEXT:   p4.store(%3, %1) : (!p4.ref<si32>, si32) -> ()
// CHECK-NEXT:   %4 = p4.get_member_ref(%2) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT:   %5 = p4.constant true
// CHECK-NEXT:   p4.store(%4, %5) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT:   %6 = p4.load(%2) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT:   p4.init !p4.header<"MyHeader"> (%6) : (!p4.header<"MyHeader">)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @chk3 : !p4.extern_class<"Checksum"<!p4.header<"MyHeader">>> {
// CHECK-NEXT:   %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:   %1 = p4.get_member_ref(%0) "hdr" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT:   %2 = p4.load(%1) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT:   p4.init !p4.extern_class<"Checksum"<!p4.header<"MyHeader">>> (%2) : (!p4.header<"MyHeader">)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @chk4 : !p4.extern_class<"Checksum"<si32>> {
// CHECK-NEXT:   %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:   %1 = p4.get_member_ref(%0) "hdr" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT:   %2 = p4.get_member_ref(%1) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si32>
// CHECK-NEXT:   %3 = p4.load(%2) : !p4.ref<si32> -> si32
// CHECK-NEXT:   p4.init !p4.extern_class<"Checksum"<si32>> (%3) : (si32)
// CHECK-NEXT: }

// CHECK-NEXT: p4.apply {
// CHECK-NEXT:   %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT:   p4.return
// CHECK-NEXT: }

// CHECK-NEXT: }
// CHECK-NEXT: }




