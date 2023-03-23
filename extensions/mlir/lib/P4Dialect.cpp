#include "P4Dialect.h"
#include "P4Ops.h"

using namespace mlir;
using namespace p4mlir;

#include "P4OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void p4mlir::P4Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "P4Ops.cpp.inc"
      >();
}
