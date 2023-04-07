#include "P4Dialect.h"
#include "P4Ops.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace p4mlir;

#define GET_TYPEDEF_CLASSES
#include "P4OpsTypes.cpp.inc"
#include "P4OpsDialect.cpp.inc"

void p4mlir::P4Dialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "P4Ops.cpp.inc"
    >();
    addTypes<
        #define GET_TYPEDEF_LIST
        #include "P4OpsTypes.cpp.inc"
    >();
}
