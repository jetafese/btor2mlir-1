#include "Conversion/BtorNDToLLVM/ConvertBtorNDToLLVMPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <string>

using namespace mlir;

#define PASS_NAME "convert-btornd-to-llvm"

namespace {

unsigned numConcats(unsigned opWidth, unsigned ndFunc = 32) {
  if (opWidth <= ndFunc) {
    return 0;
  }
  if ((opWidth % ndFunc) == 0) {
    return opWidth / ndFunc;
  }
  return (opWidth / ndFunc) + 1;
}

template <typename Op>
Value getNDValueHelper(Op op, mlir::ConversionPatternRewriter &rewriter,
                       ModuleOp module, Type resultType, unsigned ndSize = 32) {
  const std::string havoc = "nd_bv32";
  auto havocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(havoc);
  if (!havocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto havocFuncTy =
        LLVM::LLVMFunctionType::get(rewriter.getIntegerType(ndSize), {});
    havocFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                  havoc, havocFuncTy);
  }
  Value callND =
      rewriter.create<LLVM::CallOp>(op.getLoc(), havocFunc, llvm::None)
          .getResult(0);
  return callND;
}

template <typename Op>
void createPrintFunctionHelper(Op op, const Value ndValue,
                               const std::string printHelper,
                               mlir::ConversionPatternRewriter &rewriter,
                               ModuleOp module, Type resultType) {
  auto printFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(printHelper);
  auto i64Type = rewriter.getI64Type();
  if (!printFunc) {
    OpBuilder::InsertionGuard printerGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto printFuncTy = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()),
        {i64Type, i64Type, i64Type});
    printFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                  printHelper, printFuncTy);
  }
  Value ndValueWidth = rewriter.create<LLVM::ConstantOp>(
      op.getLoc(), resultType,
      rewriter.getIntegerAttr(resultType, resultType.getIntOrFloatBitWidth()));
  Value zextNDWidth =
      rewriter.create<LLVM::ZExtOp>(op.getLoc(), i64Type, ndValueWidth);
  Value ndValueId = rewriter.create<LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI64Type(), op.idAttr());
  // TODO: We need to handle values with bitwidth > 64
  auto needsExt = ndValue.getType().getIntOrFloatBitWidth() <= 64;
  if (needsExt) {
    Value zextNDValue =
        rewriter.create<LLVM::ZExtOp>(op.getLoc(), i64Type, ndValue);
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange({}), printHelper,
        ValueRange({ndValueId, zextNDValue, zextNDWidth}));
    return;
  }
}

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct NDStateOpLowering : public ConvertOpToLLVMPattern<btor::NDStateOp> {
  using ConvertOpToLLVMPattern<btor::NDStateOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::NDStateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = typeConverter->convertType(op.result().getType());
    auto module = op->getParentOfType<ModuleOp>();
    auto callND = getNDValueHelper(op, rewriter, module, opType);
    // add helper function for printing
    std::string printHelper = "btor2mlir_print_state_num";
    createPrintFunctionHelper(op, callND, printHelper, rewriter, module,
                              opType);
    rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, TypeRange({opType}), callND);
    return success();
  }
};

struct InputOpLowering : public ConvertOpToLLVMPattern<btor::InputOp> {
  using ConvertOpToLLVMPattern<btor::InputOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::InputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = typeConverter->convertType(op.result().getType());
    auto module = op->getParentOfType<ModuleOp>();
    auto callND = getNDValueHelper(op, rewriter, module, opType);
    // add helper function for printing
    std::string printHelper = "btor2mlir_print_input_num";
    createPrintFunctionHelper(op, callND, printHelper, rewriter, module,
                              opType);
    rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, TypeRange({opType}), callND);
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBTORNDTOLLVMConversionPatterns(
    BtorToLLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<NDStateOpLowering, InputOpLowering>(converter);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

struct BtorNDToLLVMLoweringPass
    : public ConvertBtorNDToLLVMBase<BtorNDToLLVMLoweringPass> {
  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    BtorToLLVMTypeConverter converter(&getContext());

    mlir::btor::populateBTORNDTOLLVMConversionPatterns(converter, patterns);

    target.addIllegalOp<btor::NDStateOp, btor::InputOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

/// Create a pass for lowering operations the remaining `Btor` operations
// to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerBtorNDToLLVMPass() {
  return std::make_unique<BtorNDToLLVMLoweringPass>();
}
