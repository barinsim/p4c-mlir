// RUN: p4c-mlir-translate %s | FileCheck %s

extern Checksum16 {
    Checksum16();
    void clear();
}

extern packet_in {
    void extract<T>(out T hdr);
    void extract<T>(out T variableSizeHeader,
                    in bit<32> variableFieldSizeInBits);
    T lookahead<T>();
    void advance(in bit<32> sizeInBits);
    bit<32> length();
}

struct Parsed_packet {
    bit<48> ethernet;
    bit<32> ip;
}

parser TopParser(packet_in b, out Parsed_packet p) {
    Checksum16() ck;
    bit<32> val1 = 2;
    Parsed_packet val2;

    state start {
        b.extract<bit<48>>(p.ethernet);
        transition parse_ipv4;
    }

    state parse_ipv4 {
        b.extract<bit<32>>(p.ip);
        // TODO: verify(p.ip.version == 4w4, error.IPv4IncorrectVersion);
        // TODO: verify(p.ip.ihl == 4w5, error.IPv4OptionsNotSupported);
        ck.clear();
        // TODO: verify(ck.get() == 16w0, error.IPv4ChecksumError);
        transition accept;
    }

    state drop {}

    state drop_explicit {
        transition reject;
    }

    state check_ssa {
        int<16> x1 = 3;
        if (x1 == 2) {
            x1 = 4;
        } else {
            x1 = 2;
        }
        x1 = x1 + 1;
    }
}

// CHECK: module {
// CHECK-NEXT: p4.extern_class @Checksum16 {
// CHECK-NEXT: p4.constructor @Checksum16_0()
// CHECK-NEXT: p4.extern @clear_0()
// CHECK-NEXT: }
// CHECK-NEXT: p4.extern_class @packet_in {
// CHECK-NEXT: p4.extern @extract_1<@T>(!p4.ref<!p4.type_var<@T>>)
// CHECK-NEXT: p4.extern @extract_2<@T>(!p4.ref<!p4.type_var<@T>>, ui32)
// CHECK-NEXT: p4.extern @lookahead_0<@T>() -> !p4.type_var<@T>
// CHECK-NEXT: p4.extern @advance_1(ui32)
// CHECK-NEXT: p4.extern @length_0() -> ui32
// CHECK-NEXT: }
// CHECK-NEXT: p4.struct @Parsed_packet {
// CHECK-NEXT: p4.member_decl @ethernet : ui48
// CHECK-NEXT: p4.member_decl @ip : ui32
// CHECK-NEXT: }
// CHECK-NEXT: p4.parser @TopParser(%arg0: !p4.ref<!p4.extern_class<"packet_in">>, %arg1: !p4.ref<!p4.struct<"Parsed_packet">>) {
// CHECK-NEXT: p4.member_decl @ck : !p4.extern_class<"Checksum16"> {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT: p4.init @Checksum16::@Checksum16_0 !p4.extern_class<"Checksum16"> () : ()
// CHECK-NEXT: }
// CHECK-NEXT: p4.state @start(%arg2: !p4.ref<ui32>, %arg3: !p4.ref<!p4.struct<"Parsed_packet">>) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT: %1 = p4.get_member_ref(%arg1) "ethernet" : !p4.ref<!p4.struct<"Parsed_packet">> -> !p4.ref<ui48>
// CHECK-NEXT: %2 = p4.alloc : !p4.ref<ui48>
// CHECK-NEXT: p4.call_method %arg0 @packet_in::@extract_1<ui48>(%2) : (!p4.ref<!p4.extern_class<"packet_in">>, !p4.ref<ui48>) -> ()
// CHECK-NEXT: %3 = p4.load(%2) : !p4.ref<ui48> -> ui48
// CHECK-NEXT: p4.store(%1, %3) : (!p4.ref<ui48>, ui48) -> ()
// CHECK-NEXT: p4.transition @TopParser::@parse_ipv4(%arg2, %arg3) : (!p4.ref<ui32>, !p4.ref<!p4.struct<"Parsed_packet">>)
// CHECK-NEXT: }
// CHECK-NEXT: p4.state @parse_ipv4(%arg2: !p4.ref<ui32>, %arg3: !p4.ref<!p4.struct<"Parsed_packet">>) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT: %1 = p4.get_member_ref(%arg1) "ip" : !p4.ref<!p4.struct<"Parsed_packet">> -> !p4.ref<ui32>
// CHECK-NEXT: %2 = p4.alloc : !p4.ref<ui32>
// CHECK-NEXT: p4.call_method %arg0 @packet_in::@extract_1<ui32>(%2) : (!p4.ref<!p4.extern_class<"packet_in">>, !p4.ref<ui32>) -> ()
// CHECK-NEXT: %3 = p4.load(%2) : !p4.ref<ui32> -> ui32
// CHECK-NEXT: p4.store(%1, %3) : (!p4.ref<ui32>, ui32) -> ()
// CHECK-NEXT: %4 = p4.get_member_ref(%0) "ck" : !p4.ref<!p4.parser<"TopParser">> -> !p4.ref<!p4.extern_class<"Checksum16">>
// CHECK-NEXT: p4.call_method %4 @Checksum16::@clear_0() : (!p4.ref<!p4.extern_class<"Checksum16">>) -> ()
// CHECK-NEXT: p4.parser_accept
// CHECK-NEXT: }
// CHECK-NEXT: p4.state @drop(%arg2: !p4.ref<ui32>, %arg3: !p4.ref<!p4.struct<"Parsed_packet">>) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT: p4.parser_reject
// CHECK-NEXT: }
// CHECK-NEXT: p4.state @drop_explicit(%arg2: !p4.ref<ui32>, %arg3: !p4.ref<!p4.struct<"Parsed_packet">>) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT: p4.parser_reject
// CHECK-NEXT: }
// CHECK-NEXT: p4.state @check_ssa(%arg2: !p4.ref<ui32>, %arg3: !p4.ref<!p4.struct<"Parsed_packet">>) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT: %1 = p4.constant 3 : si16
// CHECK-NEXT: %2 = p4.cast(%1) : si16 -> si16
// CHECK-NEXT: %3 = p4.copy(%2) : si16 -> si16
// CHECK-NEXT: %4 = p4.constant 2 : si64
// CHECK-NEXT: %5 = p4.cast(%4) : si64 -> si16
// CHECK-NEXT: %6 = p4.cmp(%3, %5) eq : (si16, si16) -> i1
// CHECK-NEXT: cf.cond_br %6, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT: %7 = p4.constant 4 : si16
// CHECK-NEXT: %8 = p4.cast(%7) : si16 -> si16
// CHECK-NEXT: %9 = p4.copy(%8) : si16 -> si16
// CHECK-NEXT: cf.br ^bb3(%9 : si16)
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT: %10 = p4.constant 2 : si16
// CHECK-NEXT: %11 = p4.cast(%10) : si16 -> si16
// CHECK-NEXT: %12 = p4.copy(%11) : si16 -> si16
// CHECK-NEXT: cf.br ^bb3(%12 : si16)
// CHECK-NEXT: ^bb3(%13: si16):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT: %14 = p4.constant 1 : si16
// CHECK-NEXT: %15 = p4.add(%13, %14) : (si16, si16) -> si16
// CHECK-NEXT: %16 = p4.copy(%15) : si16 -> si16
// CHECK-NEXT: p4.parser_reject
// CHECK-NEXT: }
// CHECK-NEXT: p4.apply {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT: %1 = p4.constant 2 : ui32
// CHECK-NEXT: %2 = p4.cast(%1) : ui32 -> ui32
// CHECK-NEXT: %3 = p4.alloc : !p4.ref<ui32>
// CHECK-NEXT: p4.store(%3, %2) : (!p4.ref<ui32>, ui32) -> ()
// CHECK-NEXT: %4 = p4.uninitialized : !p4.struct<"Parsed_packet">
// CHECK-NEXT: %5 = p4.alloc : !p4.ref<!p4.struct<"Parsed_packet">>
// CHECK-NEXT: p4.store(%5, %4) : (!p4.ref<!p4.struct<"Parsed_packet">>, !p4.struct<"Parsed_packet">) -> ()
// CHECK-NEXT: p4.transition @TopParser::@start(%3, %5) : (!p4.ref<ui32>, !p4.ref<!p4.struct<"Parsed_packet">>)
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }

