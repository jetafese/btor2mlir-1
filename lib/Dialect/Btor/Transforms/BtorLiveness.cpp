//===- BtorLiveness.cpp - Standard Function conversions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/Transforms/BtorLiveness.h"
#include "Dialect/Btor/IR/Btor.h"
#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace btor;

namespace {
bool opsMatch(Value &value, llvm::StringLiteral name) {
  if (value.isa<BlockArgument>()) {
    return false;
  }
  return value.getDefiningOp()->getName().getStringRef().equals(name);
}

bool opsMatch(Operation *op, llvm::StringLiteral name) {
  return op->getName().getStringRef().equals(name);
}

LogicalResult replaceWithWriteInPlace(btor::WriteOp &op) {
  auto status = resultIsLiveAfter(op);
  if (status.succeeded()) {
    auto opPtr = op.getOperation();
    auto m_context = opPtr->getContext();
    auto m_builder = OpBuilder(m_context);
    auto resValue = op.result();
    assert(resValue.hasOneUse());
    auto useOp = resValue.user_begin().getCurrent().getUser();
    if (opsMatch(useOp, btor::IteOp::getOperationName())) {
      // replace ite op with ite_write_in_place
      auto iteOpResult = useOp->getResult(0);
      m_builder.setInsertionPointAfterValue(iteOpResult);
      Value iteWriteInPlaceOp = m_builder.create<btor::IteWriteInPlaceOp>(
          op.getLoc(), op.getType(), useOp->getOperand(0), op.value(),
          op.base(), op.index());
      iteOpResult.replaceAllUsesWith(iteWriteInPlaceOp);
      assert(iteOpResult.use_empty());
      assert(useOp->use_empty());
      useOp->erase();
      assert(opPtr->use_empty());
    } else {
      m_builder.setInsertionPointAfterValue(resValue);
      assert(opsMatch(useOp, mlir::BranchOp::getOperationName()));
      Value writeInPlace = m_builder.create<btor::WriteInPlaceOp>(
          op.getLoc(), op.getType(), op.value(), op.base(), op.index());
      resValue.replaceAllUsesWith(writeInPlace);
      assert(resValue.use_empty());
    }
  }

  return status;
}

/// @brief Make sure all uses of baseArray are before the newArray operation
/// @param baseArray, newArray, opPtr
/// @return success/failure based on identified uses
LogicalResult moveAfterReadOps(Value &baseArray, Value &newArray,
                               Operation *opPtr) {
  for (auto it = baseArray.user_begin(); it != baseArray.user_end(); ++it) {
    auto curUse = it.getCurrent().getUser();
    if ((curUse != opPtr) && (!curUse->isBeforeInBlock(opPtr))) {
      if (!opsMatch(curUse, btor::ReadOp::getOperationName())) {
        return failure();
      }
      opPtr->moveAfter(curUse);
      newArray.getDefiningOp()->moveAfter(curUse);
      assert(curUse->isBeforeInBlock(opPtr));
      assert(curUse->isBeforeInBlock(newArray.getDefiningOp()));
    }
  }
  return success();
}

/// @brief Make sure all uses of baseArray are before the given operation
/// @param baseArray, opPtr
/// @return success/failure based on identified uses
LogicalResult moveAfterReadOps(Value &baseArray, Operation *opPtr) {
  for (auto it = baseArray.user_begin(); it != baseArray.user_end(); ++it) {
    auto curUse = it.getCurrent().getUser();
    if ((curUse != opPtr) && (!curUse->isBeforeInBlock(opPtr))) {
      if (!opsMatch(curUse, btor::ReadOp::getOperationName())) {
        return failure();
      }
      opPtr->moveAfter(curUse);
      assert(curUse->isBeforeInBlock(opPtr));
    }
  }
  return success();
}

/// find and replace ite pattern below
///  %wr = write %v1, %A[%i1]
///  %ite = ite %c1, %wr, %A
///  return %ite
LogicalResult usedInITEPattern(btor::IteOp &iteOp) {
  auto opPtr = iteOp.getOperation();
  Value trueValue = iteOp.true_value();
  Value falseValue = iteOp.false_value();

  if (opsMatch(trueValue, btor::WriteOp::getOperationName())) {
    assert(trueValue.hasOneUse());
    assert(falseValue == trueValue.getDefiningOp()->getOperand(1));
    return moveAfterReadOps(falseValue, trueValue, opPtr);
  }
  assert(opsMatch(falseValue, btor::WriteOp::getOperationName()));
  assert(falseValue.hasOneUse());
  assert(trueValue == falseValue.getDefiningOp()->getOperand(1));
  return moveAfterReadOps(trueValue, falseValue, opPtr);
}

/// find and replace ite pattern below
///  %wr = write %v1, %A[%i1]
///  return %wr
LogicalResult usedInWritePattern(btor::WriteOp &writeOp) {
  auto opPtr = writeOp.getOperation();
  Value baseArray = writeOp.base();
  return moveAfterReadOps(baseArray, opPtr);
}

struct BtorLivenessPass : public BtorLivenessBase<BtorLivenessPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    auto module_regions = rootOp->getRegions();
    auto &blocks = module_regions.front().getBlocks();
    auto &funcOp = blocks.front().getOperations().front();
    auto &regions = funcOp.getRegion(0);
    assert(regions.getBlocks().size() == 2);
    auto &nextBlock = regions.getBlocks().back();

    for (Operation &op : nextBlock.getOperations()) {
      LogicalResult status =
          llvm::TypeSwitch<Operation *, LogicalResult>(&op)
              // btor ops.
              .Case<btor::WriteOp>(
                  [&](auto op) { return replaceWithWriteInPlace(op); })
              .Default([&](Operation *) { return success(); });
      assert(status.succeeded());
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::btor::computeBtorLiveness() {
  return std::make_unique<BtorLivenessPass>();
}

/// @brief Determine if a writeOp is used by non-branch operations
/// @param Btor WriteOp
/// @return success/failure wrapped in LogicalResult
LogicalResult mlir::btor::resultIsLiveAfter(btor::WriteOp &op) {
  auto opPtr = op.getOperation();
  auto blockPtr = opPtr->getBlock();
  Value resValue = op.result();

  assert(opPtr != nullptr);
  assert(blockPtr != nullptr);
  assert(!resValue.isUsedOutsideOfBlock(blockPtr));
  assert(resValue.hasOneUse());
  if (!resValue.hasOneUse()) {
    return failure();
  }
  auto use = resValue.user_begin();
  auto useOp = use.getCurrent().getUser();
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(useOp)
          .Case<mlir::BranchOp>([&](auto op) { return success(); })
          .Case<btor::IteOp>([&](auto op) { return usedInITEPattern(op); })
          .Default([&](Operation *) { return failure(); });
  return status;
}
