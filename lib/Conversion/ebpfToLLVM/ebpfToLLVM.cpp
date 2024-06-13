#include "Conversion/ebpfToLLVM/ConvertebpfToLLVMPass.h"
#include "Dialect/ebpf/IR/ebpf.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"

#include <string>

using namespace mlir;

#define PASS_NAME "convert-ebpf-to-llvm"

namespace {

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//
#define CONVERT_OP(EBPF, LLVM) mlir::VectorConvertToLLVMPattern<EBPF, LLVM>

/** division operations will need to abort when dividing by zero **/
using AddOpLowering = CONVERT_OP(ebpf::AddOp, LLVM::AddOp);
using SubOpLowering = CONVERT_OP(ebpf::SubOp, LLVM::SubOp);
using MulOpLowering = CONVERT_OP(ebpf::MulOp, LLVM::MulOp);
using SDivOpLowering = CONVERT_OP(ebpf::SDivOp, LLVM::SDivOp);
using UDivOpLowering = CONVERT_OP(ebpf::UDivOp, LLVM::UDivOp);
using SModOpLowering = CONVERT_OP(ebpf::SModOp, LLVM::SRemOp);
using UModOpLowering = CONVERT_OP(ebpf::UModOp, LLVM::URemOp);
using OrOpLowering = CONVERT_OP(ebpf::OrOp, LLVM::OrOp);
using XOrOpLowering = CONVERT_OP(ebpf::XOrOp, LLVM::XOrOp);
using ShiftLLOpLowering = CONVERT_OP(ebpf::LSHOp, LLVM::ShlOp);
using ShiftRLOpLowering = CONVERT_OP(ebpf::RSHOp, LLVM::LShrOp);
using ShiftRAOpLowering = CONVERT_OP(ebpf::ShiftRAOp, LLVM::AShrOp);

using AndOpLowering = CONVERT_OP(ebpf::AndOp, LLVM::AndOp);

//===----------------------------------------------------------------------===//
// Op Lowerings
//===----------------------------------------------------------------------===//

// Convert ebpf.cmp predicate into the LLVM dialect CmpPredicate.
// template <typename LLVMPredType>
static LLVM::ICmpPredicate convertCmpPredicate(ebpf::ebpfPredicate pred) {
  assert(pred != ebpf::ebpfPredicate::set && "set not implemented");
  return static_cast<LLVM::ICmpPredicate>(pred);
}

struct CmpOpLowering : public ConvertOpToLLVMPattern<ebpf::CmpOp> {
  using ConvertOpToLLVMPattern<ebpf::CmpOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(ebpf::CmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto resultType = op.getResult().getType();

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, typeConverter->convertType(resultType),
        convertCmpPredicate(op.getPredicate()), adaptor.lhs(), adaptor.rhs());

    return success();
  }
};

struct ConstantOpLowering : public ConvertOpToLLVMPattern<ebpf::ConstantOp> {
  using ConvertOpToLLVMPattern<ebpf::ConstantOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ebpf::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return LLVM::detail::oneToOneRewrite(
        op, LLVM::ConstantOp::getOperationName(), adaptor.getOperands(),
        *getTypeConverter(), rewriter);
  }
};

struct NegOpLowering : public ConvertOpToLLVMPattern<ebpf::NegOp> {
  using ConvertOpToLLVMPattern<ebpf::NegOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(ebpf::NegOp negOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Value operand = adaptor.operand();
    Type opType = operand.getType();

    Value zeroConst = rewriter.create<LLVM::ConstantOp>(
        negOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 0));
    rewriter.replaceOpWithNewOp<LLVM::SubOp>(negOp, zeroConst, operand);
    return success();
  }
};

// struct SModOpLowering : public ConvertOpToLLVMPattern<ebpf::SModOp> {
//   using ConvertOpToLLVMPattern<ebpf::SModOp>::ConvertOpToLLVMPattern;
//   LogicalResult matchAndRewrite(ebpf::SModOp smodOp, OpAdaptor adaptor,
//                                 ConversionPatternRewriter &rewriter) const {
//     // since srem(a, b) = sign_of(a) * smod(a, b),
//     // we have smod(a, b) =  sign_of(b) * |srem(a, b)|
//     auto loc = smodOp.getLoc();
//     auto rhs = adaptor.rhs(), lhs = adaptor.lhs();
//     auto opType = rhs.getType();

//     Value zeroConst = rewriter.create<LLVM::ConstantOp>(
//         loc, opType, rewriter.getIntegerAttr(opType, 0));
//     Value srem = rewriter.create<btor::SRemOp>(loc, lhs, rhs);
//     Value remLessThanZero = rewriter.create<LLVM::ICmpOp>(
//         loc, LLVM::ICmpPredicate::slt, srem, zeroConst);
//     Value rhsLessThanZero = rewriter.create<LLVM::ICmpOp>(
//         loc, LLVM::ICmpPredicate::slt, rhs, zeroConst);
//     Value rhsIsNotZero = rewriter.create<LLVM::ICmpOp>(
//         loc, LLVM::ICmpPredicate::ne, rhs, zeroConst);
//     Value xorOp =
//         rewriter.create<LLVM::XOrOp>(loc, remLessThanZero, rhsLessThanZero);
//     Value needsNegationAndRhsNotZero =
//         rewriter.create<LLVM::AndOp>(loc, xorOp, rhsIsNotZero);
//     Value negOp = rewriter.create<btor::NegOp>(loc, srem);
//     rewriter.replaceOpWithNewOp<LLVM::SelectOp>(
//         smodOp, needsNegationAndRhsNotZero, negOp, srem);
//     return success();
//   }
// };

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

struct ebpfToLLVMLoweringPass
    : public ConvertebpfToLLVMBase<ebpfToLLVMLoweringPass> {

  ebpfToLLVMLoweringPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  StringRef getArgument() const final { return PASS_NAME; }
  void runOnOperation() override;
};
} // end anonymous namespace

void ebpfToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  ebpfToLLVMTypeConverter converter(&getContext(), true);

  mlir::ebpf::populateebpfToLLVMConversionPatterns(converter, patterns);
  mlir::populateStdToLLVMConversionPatterns(converter, patterns);

  /// Configure conversion to lower out ebpf; Anything else is fine.
  // /// unary operators
  target.addIllegalOp<ebpf::NegOp, ebpf::BE16, ebpf::BE32, ebpf::BE64,
                      ebpf::LE16, ebpf::LE32, ebpf::LE64, ebpf::SWAP16,
                      ebpf::SWAP32, ebpf::SWAP64, ebpf::ConstantOp>();

  /// binary operators
  // logical
  target.addIllegalOp<ebpf::CmpOp, ebpf::LSHOp, ebpf::RSHOp, ebpf::ShiftRAOp,
                      ebpf::XOrOp, ebpf::OrOp, ebpf::AndOp>();

  // arithmetic
  target.addIllegalOp<ebpf::AddOp, ebpf::SubOp, ebpf::MulOp, ebpf::SDivOp,
                      ebpf::UDivOp, ebpf::SModOp, ebpf::UModOp>();
  // , ebpf::MoveOp,
  // ebpf::Move32Op, ebpf::Move16Op, ebpf::Move8Op,
  // ebpf::LoadMapOp>();

  /// ternary operators
  // target.addIllegalOp<ebpf::StoreOp, ebpf::Store32Op, ebpf::Store16Op,
  //                     ebpf::Store8Op, ebpf::LoadOp, ebpf::Load32Op,
  //                     ebpf::Load16Op, ebpf::Load8Op>();

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::ebpf::populateebpfToLLVMConversionPatterns(
    ebpfToLLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<AddOpLowering, SubOpLowering, MulOpLowering, SModOpLowering,
               UModOpLowering, AndOpLowering, SDivOpLowering, UDivOpLowering,
               NegOpLowering, OrOpLowering, XOrOpLowering, ShiftLLOpLowering,
               ShiftRLOpLowering, ShiftRAOpLowering, CmpOpLowering,
               ConstantOpLowering>(converter);
}

/// Create a pass for lowering operations the remaining `ebpf` operations
// to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::ebpf::createLowerToLLVMPass() {
  return std::make_unique<ebpfToLLVMLoweringPass>();
}
