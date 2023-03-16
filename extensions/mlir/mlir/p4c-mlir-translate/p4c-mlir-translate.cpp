#include "ssa.h"
#include "mlir/InitAllTranslations.h"

int main() {
    mlir::registerAllTranslations();
    return 0;
}