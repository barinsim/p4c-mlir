// File "very_simple_switch_model.p4"
// Very Simple Switch P4 declaration
// core library needed for packet_in and packet_out definitions
# include <core.p4>
/* Various constants and structure declarations */
/* ports are represented using 4-bit values */
typedef bit<4> PortId;
/* only 8 ports are "real" */
const PortId REAL_PORT_COUNT = 4w8;  // 4w8 is the number 8 in 4 bits
/* metadata accompanying an input packet */
struct InControl {
    PortId inputPort;
}
/* special input port values */
const PortId RECIRCULATE_IN_PORT = 0xD;
const PortId CPU_IN_PORT = 0xE;
/* metadata that must be computed for outgoing packets */
struct OutControl {
    PortId outputPort;
}
/* special output port values for outgoing packet */
const PortId DROP_PORT = 0xF;
const PortId CPU_OUT_PORT = 0xE;
const PortId RECIRCULATE_OUT_PORT = 0xD;
/* Prototypes for all programmable blocks */
/**
 * Programmable parser.
 * @param <H> type of headers; defined by user
 * @param b input packet
 * @param parsedHeaders headers constructed by parser
 */
parser Parser<H>(packet_in b,
                 out H parsedHeaders);
/**
 * Match-action pipeline
 * @param <H> type of input and output headers
 * @param headers headers received from the parser and sent to the deparser
 * @param parseError error that may have surfaced during parsing
 * @param inCtrl information from architecture, accompanying input packet
 * @param outCtrl information for architecture, accompanying output packet
 */
control Pipe<H>(inout H headers,
                in error parseError,// parser error
                in InControl inCtrl,// input port
                out OutControl outCtrl); // output port
/**
 * VSS deparser.
 * @param <H> type of headers; defined by user
 * @param b output packet
 * @param outputHeaders headers for output packet
 */
control Deparser<H>(inout H outputHeaders,
                    packet_out b);
/**
 * Top-level package declaration - must be instantiated by user.
 * The arguments to the package indicate blocks that
 * must be instantiated by the user.
 * @param <H> user-defined type of the headers processed.
 */
package VSS<H>(Parser<H> p,
               Pipe<H> map,
               Deparser<H> d);
// Architecture-specific objects that can be instantiated
// Checksum unit
extern Checksum16 {
    Checksum16();              // constructor
    void clear();              // prepare unit for computation
    void update<T>(in T data); // add data to checksum
    void remove<T>(in T data); // remove data from existing checksum
    bit<16> get(); // get the checksum for the data added since last clear
}

// This program processes packets comprising an Ethernet and an IPv4
// header, and it forwards packets using the destination IP address

typedef bit<48>  EthernetAddress;
typedef bit<32>  IPv4Address;

// Standard Ethernet header
header Ethernet_h {
    EthernetAddress dstAddr;
    EthernetAddress srcAddr;
    bit<16>         etherType;
}

// IPv4 header (without options)
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
    IPv4Address  srcAddr;
    IPv4Address  dstAddr;
}

// Structure of parsed headers
struct Parsed_packet {
    Ethernet_h ethernet;
    IPv4_h     ip;
}

// Parser section

// User-defined errors that may be signaled during parsing
error {
    IPv4OptionsNotSupported,
    IPv4IncorrectVersion,
    IPv4ChecksumError
}

parser TopParser(packet_in b, out Parsed_packet p) {
    Checksum16() ck;  // instantiate checksum unit
    // Checksum16() ck#1;

    state start {
        b.extract(p.ethernet);
        // p.ethernet.etherType#1, b#2 = b#1.extract(size); [sideEffect] // for each field
        
        transition select(p.ethernet.etherType) {
            0x0800: parse_ipv4;
            // no default rule: all other packets rejected
        }
        // transition select(p.ethernet.etherType#1) {
        //     0x0800: parse_ipv4(/*all fields of p*/, b#2);
        //     // no default rule: all other packets rejected
        // }
    }

    state parse_ipv4(/*packet fields*/, /*packet_in b#3*/) {
        b.extract(p.ip);
        // p.ip.version#1, b#4 = b#3.extract(size); [sideEffect]
        // p.ip.ihl#1 = b.extract(size); [sideEffect]
        verify(p.ip.version == 4w4, error.IPv4IncorrectVersion);
        // verify(p.ip.version#1 == 4w4, error.IPv4IncorrectVersion);
        // FALSE: reject(/*all fields of p*/ ???)
        verify(p.ip.ihl == 4w5, error.IPv4OptionsNotSupported);
        // verify(p.ip.ihl#1 == 4w5, error.IPv4OptionsNotSupported);
        // FALSE: reject(/*all fields of p*/ ???)
        ck.clear();
        // ck#2 = ck#1.clear(); [sideEffect] // Maybe we also have to change global memory state ???
        ck.update(p.ip);
        // ck#3 = ck#2.update(p.ip /* all fields*/); [sideEffect]
        verify(ck.get() == 16w0, error.IPv4ChecksumError);
        // tmp#1 = ck#3.get() [noSideEffect]
        // verify(tmp#1 == 16w0, error.IPv4ChecksumError);
        // FALSE: reject(/*all fields of p*/ ???)
        transition accept(/*packet fields*/, /*packet_in b#3*/);
    }

    /*
    state accept/reject {
        return packet_fields
    }
    */
}

// Match-action pipeline section

control TopPipe(inout Parsed_packet headers,
                in error parseError, // parser error
                in InControl inCtrl, // input port
                out OutControl outCtrl) {
     IPv4Address nextHop;  // local variable

     /**
      * Indicates that a packet is dropped by setting the
      * output port to the DROP_PORT
      */
      // outCtrl
      action Drop_action(/*outCtrl*/) { 
          outCtrl.outputPort = DROP_PORT;
      }
      // takes all control parameters that it uses or writes to
      // returns all parameters that it writes to

     /**
      * Set the next hop and the output port.
      * Decrements ipv4 ttl field.
      * @param ivp4_dest ipv4 address of next hop
      * @param port output port
      */
      action Set_nhop(IPv4Address ipv4_dest, PortId port) {
          nextHop = ipv4_dest;
          headers.ip.ttl = headers.ip.ttl - 1;
          outCtrl.outputPort = port;
      }

     /**
      * Computes address of next IPv4 hop and output port
      * based on the IPv4 destination of the current packet.
      * Decrements packet IPv4 TTL.
      * @param nextHop IPv4 address of next hop
      */
     table ipv4_match {
         key = { headers.ip.dstAddr: lpm; }  // longest-prefix match
         actions = {
              Drop_action;
              Set_nhop;
         }
         size = 1024;
         default_action = Drop_action;
     }

     /**
      * Send the packet to the CPU port
      */
      action Send_to_cpu() {
          outCtrl.outputPort = CPU_OUT_PORT;
      }

     /**
      * Check packet TTL and send to CPU if expired.
      */
     table check_ttl {
         key = { headers.ip.ttl: exact; }
         actions = { Send_to_cpu; NoAction; }
         const default_action = NoAction; // defined in core.p4
     }

     /**
      * Set the destination MAC address of the packet
      * @param dmac destination MAC address.
      */
      action Set_dmac(EthernetAddress dmac) {
          headers.ethernet.dstAddr = dmac;
      }

     /**
      * Set the destination Ethernet address of the packet
      * based on the next hop IP address.
      * @param nextHop IPv4 address of next hop.
      */
      table dmac {
          key = { nextHop: exact; }
          actions = {
               Drop_action;
               Set_dmac;
          }
          size = 1024;
          default_action = Drop_action;
      }

      /**
       * Set the source MAC address.
       * @param smac: source MAC address to use
       */
       action Set_smac(EthernetAddress smac) {
           headers.ethernet.srcAddr = smac;
       }

      /**
       * Set the source mac address based on the output port.
       */
      table smac {
           key = { outCtrl.outputPort: exact; }
           actions = {
                Drop_action;
                Set_smac;
          }
          size = 16;
          default_action = Drop_action;
      }

      apply {
          if (parseError != error.NoError) {
              Drop_action();  // invoke drop directly
              return;
          }

          ipv4_match.apply(); // Match result will go into nextHop
          // first tmp = match(keys)
          // then switch(tmp) { case: h.f1 = act1(), case: nexthop = act2(nexthop) }
          if (outCtrl.outputPort == DROP_PORT) return;

          check_ttl.apply();
          if (outCtrl.outputPort == CPU_OUT_PORT) return;

          dmac.apply();
          if (outCtrl.outputPort == DROP_PORT) return;

          smac.apply();
          
          // return packet_fields, out_ctrl
    }
}

// deparser section
control TopDeparser(inout Parsed_packet p, packet_out b) {
    Checksum16() ck;
    apply {
        b.emit(p.ethernet);
        if (p.ip.isValid()) {
            ck.clear();              // prepare checksum unit
            p.ip.hdrChecksum = 16w0; // clear checksum
            ck.update(p.ip);         // compute new checksum.
            p.ip.hdrChecksum = ck.get();
        }
        b.emit(p.ip);
    }
}

// Instantiate the top-level VSS package
VSS(TopParser(),
    TopPipe(),
    TopDeparser()) main;