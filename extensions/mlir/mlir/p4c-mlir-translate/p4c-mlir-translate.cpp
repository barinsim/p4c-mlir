#include "ssa.h"
#include "mlir/InitAllTranslations.h"
#include "P4Dialect.h"
#include "P4Ops.h"

int main() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<p4mlir::P4Dialect>();
    return 0;
}