#include "Conversion/BtorToMemref/ConvertBtorToMemrefPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

using namespace mlir;
using namespace mlir::btor;

void mlir::btor::populateBtorToMemrefConversionPatterns(
    BtorToLLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // std::cout << "Hello World\n" << std::endl;
}

namespace {
struct ConvertBtorToMemrefPass
    : public ConvertBtorToMemrefBase<ConvertBtorToMemrefPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    BtorToLLVMTypeConverter converter(&getContext());

    mlir::btor::populateBtorToMemrefConversionPatterns(converter, patterns);
    // mlir::populateStdToLLVMConversionPatterns(converter, patterns);
    /// Configure conversion to lower out btor; Anything else is fine.
    // init operators
    target.addIllegalOp<btor::InitArrayOp, btor::VectorInitArrayOp>();

    /// indexed operators
    target.addIllegalOp<btor::ReadOp, btor::VectorReadOp>();
    target.addIllegalOp<btor::WriteOp, btor::VectorWriteOp>();
    target.addIllegalOp<btor::WriteInPlaceOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::btor::createLowerToMemrefPass() {
  return std::make_unique<ConvertBtorToMemrefPass>();
}