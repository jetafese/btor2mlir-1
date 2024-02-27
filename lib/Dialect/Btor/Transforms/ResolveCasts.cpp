//===- BtorLiveness.cpp - Standard Function conversions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/Transforms/ResolveCasts.h"
#include "Dialect/Btor/IR/Btor.h"
#include "PassDetail.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/StandardOps/IR/Ops.h"
// #include "mlir/Transforms/DialectConversion.h"
// #include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace btor;

namespace {
bool isCastOp(Operation &op) {
  return isa<mlir::UnrealizedConversionCastOp>(op);
}

bool isCastOp(Operation *op) {
  return isa<mlir::UnrealizedConversionCastOp>(op);
}

bool usedByCastOp(Value &use) {
  auto user = use.getUses().begin().getUser();
  assert(user->getResults().size() == 1);
  return isCastOp(user);
}

/// @brief Resolve converision casts starting from root
/// @param root
/// @return void
void resolveFromRoot(mlir::UnrealizedConversionCastOp &root) {
  assert(root.getResults().size() == 1);
  assert(root.getOperands().size() == 1);
  std::vector<Value> casts;
  Value rootVal = root.getResult(0);
  Value source = root.getOperand(0);
  if (!rootVal.hasOneUse()) { return; }

  while(rootVal.hasOneUse() && usedByCastOp(rootVal)) {
    rootVal.dump();
    casts.push_back(rootVal);
    auto cur = rootVal.getUses().begin().getUser();
    rootVal = cur->getResult(0);
  }
  rootVal.replaceAllUsesWith(source);
  assert(rootVal.use_empty());
  rootVal.dump();
  rootVal.getDefiningOp()->erase();
  for (auto it = casts.rbegin(); it != casts.rend(); ++it){
    assert((*it).use_empty());
    (*it).getDefiningOp()->erase();
  }
}

struct ResolveCastsPass : public ResolveCastsBase<ResolveCastsPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    auto module_regions = rootOp->getRegions();
    // get the main function
    auto &blocks = module_regions.back().getBlocks();
    auto &mainFunc = blocks.front().getOperations().back();
    // get nextBlock: second block of main
    auto &regions = mainFunc.getRegion(0);
    auto &funcBlocks = regions.getBlocks();
    assert(funcBlocks.size() > 1);
    auto it = funcBlocks.begin(); ++it;
    auto &nextBlock = *it;
    // resolve all chains of unresolved casts
    unsigned rescanAfterChanges = 1;
    for (unsigned i = 0; i < rescanAfterChanges; ++i) {
      // if an unresolved cast exists, it is the first op
      auto &op = nextBlock.getOperations().front();
        if (isCastOp(op)) {
          auto castOp = cast<mlir::UnrealizedConversionCastOp>(op);
          resolveFromRoot(castOp);
          rescanAfterChanges++;
        }
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::btor::resolveCasts() {
  return std::make_unique<ResolveCastsPass>();
}

// /// @brief Determine if a writeOp is used by non-branch operations
// /// @param Btor WriteOp
// /// @return success/failure wrapped in LogicalResult
// LogicalResult mlir::btor::resultIsLiveAfter(btor::WriteOp &op) {
//   auto opPtr = op.getOperation();
//   auto blockPtr = opPtr->getBlock();
//   Value resValue = op.result();
//   if (resValue.use_empty()) {
//     return failure();
//   }

//   assert(opPtr != nullptr);
//   assert(blockPtr != nullptr);
//   assert(!resValue.isUsedOutsideOfBlock(blockPtr));
//   assert(resValue.hasOneUse());
//   if (!resValue.hasOneUse()) {
//     return failure();
//   }
//   auto use = resValue.user_begin();
//   auto useOp = use.getCurrent().getUser();
//   LogicalResult status =
//       llvm::TypeSwitch<Operation *, LogicalResult>(useOp)
//           .Case<mlir::BranchOp>(
//               [&](auto brOp) { return usedInWritePattern(op); })
//           .Case<btor::IteOp>([&](auto op) { return usedInITEPattern(op); })
//           .Default([&](Operation *) { return failure(); });
//   return status;
// }
