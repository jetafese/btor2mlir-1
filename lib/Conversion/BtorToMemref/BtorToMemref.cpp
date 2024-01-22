#include "Conversion/BtorToMemref/ConvertBtorToMemrefPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

// #include "iostream"
using namespace mlir;
using namespace mlir::btor;

void mlir::btor::populateBtorToMemrefConversionPatterns(BtorToLLVMTypeConverter &converter,
                                            RewritePatternSet &patterns)
{
    // std::cout << "Hello World\n" << std::endl;
}

namespace {
struct ConvertBtorToMemrefPass
    : public ConvertBtorToMemrefBase<ConvertBtorToMemrefPass> {
        void runOnOperation() override {
            std::cout << "Hello World" << std::endl;
        }
    
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::btor::createLowerToMemrefPass() {
  return std::make_unique<ConvertBtorToMemrefPass>();
}