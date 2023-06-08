// RUN: p4c-mlir-translate %s | FileCheck %s

error {
    NoError,
    PacketTooShort,
    NoMatch,
    StackOutOfBounds,
    HeaderTooShort,
    ParserTimeout
}

extern void verify(in bool condition, error err);

extern Checksum16 {
    Checksum16();
    void clear();
    bit<16> get();
}

extern packet_in {
    void extract<T>(out T hdr);
    void extract<T>(out T variableSizeHeader,
                    in bit<32> variableFieldSizeInBits);
    T lookahead<T>();
    void advance(in bit<32> sizeInBits);
    bit<32> length();
}

header Ethernet_h {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16>         etherType;
}

header IPv4_h {
    bit<4>       version;
    bit<4>       ihl;
    bit<8>       diffserv;
    bit<16>      totalLen;
    bit<16>      identification;
    bit<3>       flags;
    bit<13>      fragOffset;
    bit<8>       ttl;
    bit<8>       protocol;
    bit<16>      hdrChecksum;
    bit<32>  srcAddr;
    bit<32>  dstAddr;
}

struct Parsed_packet {
    Ethernet_h ethernet;
    IPv4_h ip;
}

parser TopParser(packet_in b, out Parsed_packet p) {
    Checksum16() ck;

    state start {
        b.extract<Ethernet_h>(p.ethernet);
        b.extract<IPv4_h>(p.ip);
        verify(p.ip.version == 4w4, error.ParserTimeout);
        verify(p.ip.ihl == 4w5, error.HeaderTooShort);
        ck.clear();
        verify(ck.get() == 16w0, error.PacketTooShort);
        transition accept;
    }

    state check_ssa {
        error local_err = error.PacketTooShort;
        if (local_err != error.PacketTooShort) {
            local_err = error.ParserTimeout;
        } else {
            local_err = error.NoError;
        }
        local_err = local_err;
    }
}

// CHECK: module {
// CHECK-NEXT:   p4.error @NoError
// CHECK-NEXT:   p4.error @PacketTooShort
// CHECK-NEXT:   p4.error @NoMatch
// CHECK-NEXT:   p4.error @StackOutOfBounds
// CHECK-NEXT:   p4.error @HeaderTooShort
// CHECK-NEXT:   p4.error @ParserTimeout
// CHECK-NEXT:   p4.extern @verify_2(i1, !p4.error)
// CHECK-NEXT:   p4.extern_class @Checksum16 {
// CHECK-NEXT:     p4.constructor @Checksum16_0()
// CHECK-NEXT:     p4.extern @clear_0()
// CHECK-NEXT:     p4.extern @get_0() -> ui16
// CHECK-NEXT:   }
// CHECK-NEXT:   p4.extern_class @packet_in {
// CHECK-NEXT:     p4.extern @extract_1<@T>(!p4.ref<!p4.type_var<@T>>)
// CHECK-NEXT:     p4.extern @extract_2<@T>(!p4.ref<!p4.type_var<@T>>, ui32)
// CHECK-NEXT:     p4.extern @lookahead_0<@T>() -> !p4.type_var<@T>
// CHECK-NEXT:     p4.extern @advance_1(ui32)
// CHECK-NEXT:     p4.extern @length_0() -> ui32
// CHECK-NEXT:   }
// CHECK-NEXT:   p4.header @Ethernet_h {
// CHECK-NEXT:     p4.member_decl @dstAddr : ui48
// CHECK-NEXT:     p4.member_decl @srcAddr : ui48
// CHECK-NEXT:     p4.member_decl @etherType : ui16
// CHECK-NEXT:     p4.valid_bit_decl @__valid : i1
// CHECK-NEXT:   }
// CHECK-NEXT:   p4.header @IPv4_h {
// CHECK-NEXT:     p4.member_decl @version : ui4
// CHECK-NEXT:     p4.member_decl @ihl : ui4
// CHECK-NEXT:     p4.member_decl @diffserv : ui8
// CHECK-NEXT:     p4.member_decl @totalLen : ui16
// CHECK-NEXT:     p4.member_decl @identification : ui16
// CHECK-NEXT:     p4.member_decl @flags : ui3
// CHECK-NEXT:     p4.member_decl @fragOffset : ui13
// CHECK-NEXT:     p4.member_decl @ttl : ui8
// CHECK-NEXT:     p4.member_decl @protocol : ui8
// CHECK-NEXT:     p4.member_decl @hdrChecksum : ui16
// CHECK-NEXT:     p4.member_decl @srcAddr : ui32
// CHECK-NEXT:     p4.member_decl @dstAddr : ui32
// CHECK-NEXT:     p4.valid_bit_decl @__valid : i1
// CHECK-NEXT:   }
// CHECK-NEXT:   p4.struct @Parsed_packet {
// CHECK-NEXT:     p4.member_decl @ethernet : !p4.header<"Ethernet_h">
// CHECK-NEXT:     p4.member_decl @ip : !p4.header<"IPv4_h">
// CHECK-NEXT:   }
// CHECK-NEXT:   p4.parser @TopParser(%arg0: !p4.ref<!p4.extern_class<"packet_in">>, %arg1: !p4.ref<!p4.struct<"Parsed_packet">>) {
// CHECK-NEXT:     p4.member_decl @ck : !p4.extern_class<"Checksum16"> {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       p4.init @Checksum16::@Checksum16_0 !p4.extern_class<"Checksum16"> () : ()
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @start() {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.get_member_ref(%arg1) "ethernet" : !p4.ref<!p4.struct<"Parsed_packet">> -> !p4.ref<!p4.header<"Ethernet_h">>
// CHECK-NEXT:       %2 = p4.alloc : !p4.ref<!p4.header<"Ethernet_h">>
// CHECK-NEXT:       p4.call_method %arg0 @packet_in::@extract_1<!p4.header<"Ethernet_h">>(%2) : (!p4.ref<!p4.extern_class<"packet_in">>, !p4.ref<!p4.header<"Ethernet_h">>) -> ()
// CHECK-NEXT:       %3 = p4.load(%2) : !p4.ref<!p4.header<"Ethernet_h">> -> !p4.header<"Ethernet_h">
// CHECK-NEXT:       p4.store(%1, %3) : (!p4.ref<!p4.header<"Ethernet_h">>, !p4.header<"Ethernet_h">) -> ()
// CHECK-NEXT:       %4 = p4.get_member_ref(%arg1) "ip" : !p4.ref<!p4.struct<"Parsed_packet">> -> !p4.ref<!p4.header<"IPv4_h">>
// CHECK-NEXT:       %5 = p4.alloc : !p4.ref<!p4.header<"IPv4_h">>
// CHECK-NEXT:       p4.call_method %arg0 @packet_in::@extract_1<!p4.header<"IPv4_h">>(%5) : (!p4.ref<!p4.extern_class<"packet_in">>, !p4.ref<!p4.header<"IPv4_h">>) -> ()
// CHECK-NEXT:       %6 = p4.load(%5) : !p4.ref<!p4.header<"IPv4_h">> -> !p4.header<"IPv4_h">
// CHECK-NEXT:       p4.store(%4, %6) : (!p4.ref<!p4.header<"IPv4_h">>, !p4.header<"IPv4_h">) -> ()
// CHECK-NEXT:       %7 = p4.get_member_ref(%arg1) "ip" : !p4.ref<!p4.struct<"Parsed_packet">> -> !p4.ref<!p4.header<"IPv4_h">>
// CHECK-NEXT:       %8 = p4.get_member_ref(%7) "version" : !p4.ref<!p4.header<"IPv4_h">> -> !p4.ref<ui4>
// CHECK-NEXT:       %9 = p4.load(%8) : !p4.ref<ui4> -> ui4
// CHECK-NEXT:       %10 = p4.constant 4 : ui4
// CHECK-NEXT:       %11 = p4.cmp(%9, %10) eq : (ui4, ui4) -> i1
// CHECK-NEXT:       %12 = p4.constant @ParserTimeout : !p4.error
// CHECK-NEXT:       p4.call @verify_2(%11, %12) : (i1, !p4.error) -> ()
// CHECK-NEXT:       %13 = p4.get_member_ref(%arg1) "ip" : !p4.ref<!p4.struct<"Parsed_packet">> -> !p4.ref<!p4.header<"IPv4_h">>
// CHECK-NEXT:       %14 = p4.get_member_ref(%13) "ihl" : !p4.ref<!p4.header<"IPv4_h">> -> !p4.ref<ui4>
// CHECK-NEXT:       %15 = p4.load(%14) : !p4.ref<ui4> -> ui4
// CHECK-NEXT:       %16 = p4.constant 5 : ui4
// CHECK-NEXT:       %17 = p4.cmp(%15, %16) eq : (ui4, ui4) -> i1
// CHECK-NEXT:       %18 = p4.constant @HeaderTooShort : !p4.error
// CHECK-NEXT:       p4.call @verify_2(%17, %18) : (i1, !p4.error) -> ()
// CHECK-NEXT:       %19 = p4.get_member_ref(%0) "ck" : !p4.ref<!p4.parser<"TopParser">> -> !p4.ref<!p4.extern_class<"Checksum16">>
// CHECK-NEXT:       p4.call_method %19 @Checksum16::@clear_0() : (!p4.ref<!p4.extern_class<"Checksum16">>) -> ()
// CHECK-NEXT:       %20 = p4.get_member_ref(%0) "ck" : !p4.ref<!p4.parser<"TopParser">> -> !p4.ref<!p4.extern_class<"Checksum16">>
// CHECK-NEXT:       %21 = p4.call_method %20 @Checksum16::@get_0() : (!p4.ref<!p4.extern_class<"Checksum16">>) -> ui16
// CHECK-NEXT:       %22 = p4.constant 0 : ui16
// CHECK-NEXT:       %23 = p4.cmp(%21, %22) eq : (ui16, ui16) -> i1
// CHECK-NEXT:       %24 = p4.constant @PacketTooShort : !p4.error
// CHECK-NEXT:       p4.call @verify_2(%23, %24) : (i1, !p4.error) -> ()
// CHECK-NEXT:       p4.parser_accept
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @check_ssa() {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.constant @PacketTooShort : !p4.error
// CHECK-NEXT:       %2 = p4.copy(%1) : !p4.error -> !p4.error
// CHECK-NEXT:       %3 = p4.constant @PacketTooShort : !p4.error
// CHECK-NEXT:       %4 = p4.cmp(%2, %3) ne : (!p4.error, !p4.error) -> i1
// CHECK-NEXT:       cf.cond_br %4, ^bb1, ^bb2
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %5 = p4.constant @ParserTimeout : !p4.error
// CHECK-NEXT:       %6 = p4.copy(%5) : !p4.error -> !p4.error
// CHECK-NEXT:       cf.br ^bb3(%6 : !p4.error)
// CHECK-NEXT:     ^bb2:  // pred: ^bb0
// CHECK-NEXT:       %7 = p4.constant @NoError : !p4.error
// CHECK-NEXT:       %8 = p4.copy(%7) : !p4.error -> !p4.error
// CHECK-NEXT:       cf.br ^bb3(%8 : !p4.error)
// CHECK-NEXT:     ^bb3(%9: !p4.error):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:       %10 = p4.copy(%9) : !p4.error -> !p4.error
// CHECK-NEXT:       p4.parser_reject
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.apply {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       p4.transition @TopParser::@start() : ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }



