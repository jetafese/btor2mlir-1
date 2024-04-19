#include "Conversion/BtorToMemref/ConvertBtorToMemrefPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::btor;

namespace {
LogicalResult shouldConvertToVector(Type &arrayType) {
  if (!arrayType.isa<MemRefType>()) {
    assert(arrayType.isa<VectorType>());
    return success(); /// the Vector pass will deal with this operation
  }
  return failure();
}

template <typename Op>
void createPrintFunctionHelper(Op op, const Value ndValue, const int64_t index,
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
        {i64Type, i64Type, i64Type, i64Type});
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
  Value indexInArray = rewriter.create<LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(index));
  // TODO: We need to handle values with bitwidth > 64
  auto needsExt = ndValue.getType().getIntOrFloatBitWidth() <= 64;
  if (needsExt) {
    Value zextNDValue =
        rewriter.create<LLVM::ZExtOp>(op.getLoc(), i64Type, ndValue);
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange({}), printHelper,
        ValueRange({ndValueId, indexInArray, zextNDValue, zextNDWidth}));
    return;
  }
}

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
  auto loc = op.getLoc();
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
      rewriter.create<LLVM::CallOp>(loc, havocFunc, llvm::None).getResult(0);
  // Type resultWidthType = rewriter.getI32Type();
  // unsigned resultWidth = ndSize;
  // // concatenate as many 32bit integers as needed
  auto concats = numConcats(resultType.getIntOrFloatBitWidth());
  Value finalVal;
  if (concats == 0) {
    finalVal =
        rewriter.create<LLVM::TruncOp>(loc, TypeRange({resultType}), callND);
  } else {
    finalVal = rewriter.create<LLVM::ZExtOp>(loc, resultType, callND);
  }
  return finalVal;
  // for(unsigned i = 0; i < concats; ++i) {
  //   resultWidthType = rewriter.getIntegerType(resultWidth + ndSize);
  //   resultWidth += ndSize;
  //   Value nextND = rewriter.create<LLVM::CallOp>(loc, havocFunc,
  //   llvm::None).getResult(0); Value callNDZExt =
  //     rewriter.create<LLVM::ZExtOp>(loc, resultWidthType, callND);
  //   Value shiftValue = rewriter.create<LLVM::ConstantOp>(
  //     loc, resultWidthType, rewriter.getIntegerAttr(resultWidthType,
  //     ndSize));
  //   Value lhsShiftLeft =
  //     rewriter.create<LLVM::ShlOp>(loc, callNDZExt, shiftValue);
  //   Value rhsZeroExtend =
  //     rewriter.create<LLVM::ZExtOp>(loc, resultWidthType, nextND);
  //   callND = rewriter.create<LLVM::OrOp>(loc, lhsShiftLeft, rhsZeroExtend);
  // }
  // auto finalVal = rewriter.create<LLVM::TruncOp>(loc,
  // TypeRange({resultType}), callND); return finalVal;
}

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct ArrayOpLowering : public ConvertOpToLLVMPattern<mlir::btor::ArrayOp> {
  using ConvertOpToLLVMPattern<mlir::btor::ArrayOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::ArrayOp arrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayType = typeConverter->convertType(arrayOp.getType());
    if (shouldConvertToVector(arrayType).succeeded()) {
      return success();
    }
    auto memType = arrayType.cast<MemRefType>();
    auto module = arrayOp->getParentOfType<ModuleOp>();
    auto loc = arrayOp.getLoc();
    auto newArrayOp = rewriter.create<memref::AllocOp>(loc, memType);
    auto newArray = newArrayOp.getResult();
    int64_t shape = memType.getShape().front();
    for (int64_t i = 0; i < shape; ++i) {
      auto idx = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(i), rewriter.getIndexType());
      auto callND =
          getNDValueHelper(arrayOp, rewriter, module, memType.getElementType());
      rewriter.create<memref::StoreOp>(loc, callND, newArray,
                                       ValueRange({idx}));
      std::string printHelper = "btor2mlir_print_array_state_num";
      createPrintFunctionHelper(arrayOp, callND, i, printHelper, rewriter,
                                module, memType.getElementType());
    }
    rewriter.replaceOp(arrayOp, newArray);
    return success();
  }
};

struct InitArrayLowering
    : public ConvertOpToLLVMPattern<mlir::btor::InitArrayOp> {
  using ConvertOpToLLVMPattern<mlir::btor::InitArrayOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::InitArrayOp initArrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayType = typeConverter->convertType(initArrayOp.getType());
    if (shouldConvertToVector(arrayType).succeeded()) {
      return success();
    }
    auto memType = arrayType.cast<MemRefType>();
    auto loc = initArrayOp.getLoc();
    auto newArrayOp = rewriter.create<memref::AllocOp>(loc, memType);
    auto newArray = newArrayOp.getResult();
    int64_t shape = memType.getShape().front();
    for (int64_t i = 0; i < shape; ++i) {
      auto idx = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(i), rewriter.getIndexType());
      rewriter.create<memref::StoreOp>(loc, adaptor.init(), newArray,
                                       ValueRange({idx}));
    }
    rewriter.replaceOp(initArrayOp, newArray);
    return success();
  }
};

struct ReadOpLowering : public ConvertOpToLLVMPattern<mlir::btor::ReadOp> {
  using ConvertOpToLLVMPattern<mlir::btor::ReadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::ReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = typeConverter->convertType(readOp.result().getType());
    auto arrayType = typeConverter->convertType(readOp.base().getType());
    if (shouldConvertToVector(arrayType).succeeded()) {
      return success();
    }
    rewriter.replaceOpWithNewOp<btor::MemRefReadOp>(
        readOp, resType, adaptor.base(), adaptor.index());
    return success();
  }
};

struct MemRefReadOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::MemRefReadOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::MemRefReadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::MemRefReadOp memReadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value index = rewriter.create<arith::IndexCastOp>(
        memReadOp.getLoc(), rewriter.getIndexType(), adaptor.index());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        memReadOp, memReadOp.getResult().getType(), memReadOp.base(),
        ValueRange(index));
    return success();
  }
};

struct WriteInPlaceOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::WriteInPlaceOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::WriteInPlaceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::WriteInPlaceOp writeInPlaceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType =
        typeConverter->convertType(writeInPlaceOp.result().getType());
    if (shouldConvertToVector(resType).succeeded()) {
      return success();
    }
    rewriter.replaceOpWithNewOp<btor::MemRefWriteOp>(
        writeInPlaceOp, resType, adaptor.value(), adaptor.base(),
        adaptor.index());
    return success();
  }
};

struct MemRefWriteOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::MemRefWriteOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::MemRefWriteOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::MemRefWriteOp memWriteOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value index = rewriter.create<arith::IndexCastOp>(
        memWriteOp.getLoc(), rewriter.getIndexType(), adaptor.index());
    rewriter.create<memref::StoreOp>(rewriter.getUnknownLoc(), adaptor.value(),
                                     memWriteOp.base(), ValueRange(index));
    rewriter.replaceOp(memWriteOp, memWriteOp.base());
    return success();
  }
};

struct WriteOpLowering : public ConvertOpToLLVMPattern<mlir::btor::WriteOp> {
  using ConvertOpToLLVMPattern<mlir::btor::WriteOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::WriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    /// we are working under the assumption that all patterns are
    /// identified by the liveness analysis
    assert(writeOp.use_empty() && "include the liveness analysis pass");
    writeOp.erase();
    return success();
  }
};

struct IteWriteInPlaceOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::IteWriteInPlaceOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::IteWriteInPlaceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::IteWriteInPlaceOp iteWriteInPlaceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType =
        typeConverter->convertType(iteWriteInPlaceOp.result().getType());
    if (shouldConvertToVector(resType).succeeded()) {
      return success();
    }
    auto valType =
        typeConverter->convertType(iteWriteInPlaceOp.value().getType());
    auto loc = iteWriteInPlaceOp.getLoc();
    Value curValue = rewriter.create<btor::MemRefReadOp>(
        loc, valType, adaptor.base(), adaptor.index());
    auto selectedVal = rewriter.create<LLVM::SelectOp>(
        loc, valType, adaptor.condition(), adaptor.value(), curValue);

    rewriter.create<btor::MemRefWriteOp>(loc, resType, selectedVal,
                                         adaptor.base(), adaptor.index());

    rewriter.replaceOp(iteWriteInPlaceOp, iteWriteInPlaceOp.base());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToMemrefConversionPatterns(
    BtorToLLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ReadOpLowering, WriteOpLowering, InitArrayLowering,
               MemRefReadOpLowering, MemRefWriteOpLowering, ArrayOpLowering,
               WriteInPlaceOpLowering, IteWriteInPlaceOpLowering>(converter);
}

namespace {
struct ConvertBtorToMemrefPass
    : public ConvertBtorToMemrefBase<ConvertBtorToMemrefPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    BtorToLLVMTypeConverter converter(&getContext());

    mlir::btor::populateBtorToMemrefConversionPatterns(converter, patterns);
    /// Configure conversion to lower out btor; Anything else is fine.
    // init operators
    target.addIllegalOp<btor::InitArrayOp, btor::ArrayOp>();

    // /// indexed operators
    target.addIllegalOp<btor::ReadOp, btor::MemRefReadOp>();
    target.addIllegalOp<btor::WriteOp, btor::MemRefWriteOp>();
    target.addIllegalOp<btor::WriteInPlaceOp, btor::IteWriteInPlaceOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

/// Create a pass for lowering operations Btor operations
// to the Memref dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerToMemrefPass() {
  return std::make_unique<ConvertBtorToMemrefPass>();
}