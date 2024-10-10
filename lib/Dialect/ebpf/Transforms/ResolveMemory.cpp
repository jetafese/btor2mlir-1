//===- BtorLiveness.cpp - Standard Function conversions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/ebpf/Transforms/ResolveMemory.h"
#include "Dialect/ebpf/IR/ebpf.h"
#include "PassDetail.h"

#include "llvm/ADT/TypeSwitch.h"

#include <iostream>

using namespace mlir;
using namespace ebpf;

namespace {
/// @brief Resolve ebpf memory loads when loading an address
/// @param loadOp the root operation that needs to be replaced
/// @return success if we replace a load address with loadAddr
template <typename loadOp>
LogicalResult replaceLoadWithLoadAddress(loadOp &op) {
  bool expectAddr = false, expectInt = false;
  bool replaceWithMem = true, keepAsIs = false;
  if (op.getResult().use_empty()) {
    return success();
  }
  assert(!op.getResult().use_empty());
  for (auto &use : op.getResult().getUses()) {
    expectAddr = false, expectInt = false;
    if (isa<ebpf::StoreOp>(use.getOwner())) {
      auto temp = cast<ebpf::StoreOp>(use.getOwner());
      expectAddr = (temp.lhs() == op.getResult());
      expectInt = (temp.lhs() != op.getResult());
      // std::cerr << "************" << std::endl;
    } else if (isa<ebpf::Store32Op>(use.getOwner())) {
      auto temp = cast<ebpf::Store32Op>(use.getOwner());
      expectAddr = (temp.lhs() == op.getResult());
      expectInt = (temp.lhs() != op.getResult());
      // std::cerr << "************" << std::endl;
    } else if (isa<ebpf::Store16Op>(use.getOwner())) {
      auto temp = cast<ebpf::Store16Op>(use.getOwner());
      expectAddr = (temp.lhs() == op.getResult());
      expectInt = (temp.lhs() != op.getResult());
      // std::cerr << "************" << std::endl;
    } else if (isa<ebpf::Store8Op>(use.getOwner())) {
      auto temp = cast<ebpf::Store8Op>(use.getOwner());
      expectAddr = (temp.lhs() == op.getResult());
      expectInt = (temp.lhs() != op.getResult());
      // std::cerr << "************" << std::endl;
    } else if (isa<ebpf::LoadOp>(use.getOwner())) {
      auto temp = cast<ebpf::LoadOp>(use.getOwner());
      expectAddr = (temp.lhs() == op.getResult());
      expectInt = (temp.lhs() != op.getResult());
      // std::cerr << "************" << std::endl;
    } else if (isa<ebpf::Load32Op>(use.getOwner())) {
      auto temp = cast<ebpf::Load32Op>(use.getOwner());
      expectAddr = (temp.lhs() == op.getResult());
      expectInt = (temp.lhs() != op.getResult());
      // std::cerr << "************" << std::endl;
    } else if (isa<ebpf::Load16Op>(use.getOwner())) {
      auto temp = cast<ebpf::Load16Op>(use.getOwner());
      expectAddr = (temp.lhs() == op.getResult());
      expectInt = (temp.lhs() != op.getResult());
      // std::cerr << "************" << std::endl;
    } else if (isa<ebpf::Load8Op>(use.getOwner())) {
      auto temp = cast<ebpf::Load8Op>(use.getOwner());
      expectAddr = (temp.lhs() == op.getResult());
      expectInt = (temp.lhs() != op.getResult());
      // std::cerr << "************" << std::endl;
    } else {
      expectInt = true;
      expectAddr = false;
      // std::cerr << "xxxxxxxxxxxx" << std::endl;
    }
    assert((expectAddr && !expectInt) || (!expectAddr && expectInt));
    replaceWithMem = expectAddr && replaceWithMem;
    keepAsIs = expectInt || keepAsIs;
    assert((replaceWithMem && !expectInt) || (!replaceWithMem && expectInt));
  }
  if (keepAsIs) {
    return failure();
  }
  assert(isa<ebpf::LoadOp>(op));
  if (isa<ebpf::LoadOp>(op)) {
    ebpf::LoadOp temp = cast<ebpf::LoadOp>(op);
    auto opPtr = temp.getOperation();
    auto m_context = opPtr->getContext();
    auto m_builder = OpBuilder(m_context);
    auto opResult = temp.result();
    m_builder.setInsertionPointAfterValue(opResult);
    Value loadAddr = m_builder.create<ebpf::LoadAddrOp>(
        temp.getLoc(), temp.getType(), temp.lhs(), temp.rhs());
    opResult.replaceAllUsesWith(loadAddr);
    assert(opResult.use_empty());
  }
  return success();
}

/// @brief Identify and replace loads of address with loadAddr
/// @tparam expects to receive a file with xdp_entry inlined into main
/// @return none (transformed mlir)
void processLoadAddrPattern(mlir::Operation &funcOp) {
  auto &regions = funcOp.getRegion(0);
  for (auto &block : regions.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      LogicalResult _ = llvm::TypeSwitch<Operation *, LogicalResult>(&op)
                            // ebpf ops.
                            .Case<ebpf::LoadOp>([&](auto op) {
                              return replaceLoadWithLoadAddress(op);
                            })
                            .Case<ebpf::Load32Op>([&](auto op) {
                              return replaceLoadWithLoadAddress(op);
                            })
                            .Case<ebpf::Load16Op>([&](auto op) {
                              return replaceLoadWithLoadAddress(op);
                            })
                            .Case<ebpf::Load8Op>([&](auto op) {
                              return replaceLoadWithLoadAddress(op);
                            })
                            .Default([&](Operation *) { return failure(); });
    }
  }
}

/// @brief Resolve ebpf packet accesses
/// @param op LoadPktPtrOp
/// @param pkt allocated packet
/// @return success if we replace a load address with loadAddr
LogicalResult replaceLoadPktPtrWithGep(ebpf::LoadPktPtrOp &op,
                                       ebpf::AllocaOp &pkt) {
  if (op.getResult().use_empty()) {
    return success();
  }
  ebpf::LoadPktPtrOp temp = cast<ebpf::LoadPktPtrOp>(op);
  auto opPtr = temp.getOperation();
  auto m_context = opPtr->getContext();
  auto m_builder = OpBuilder(m_context);
  auto opResult = temp.result();
  m_builder.setInsertionPointAfterValue(opResult);
  Value getAddr = m_builder.create<ebpf::GetAddrOp>(
      temp.getLoc(), temp.getType(), pkt, temp.operand());
  opResult.replaceAllUsesWith(getAddr);
  assert(opResult.use_empty());
  return success();
}

/// @brief Identify LoadPktPtrOp and replace with an appropriate gep
/// @param funcOp expects to receive a file with xdp_entry inlined into main
void adjustLoadPktPtrOps(mlir::Operation &funcOp) {
  auto &regions = funcOp.getRegion(0);
  // get operations of the first block in inlined main function
  auto &firstBlockOps = regions.getBlocks().front().getOperations();
  auto pktAllocation = firstBlockOps.front().getNextNode();
  assert(isa<ebpf::AllocaOp>(pktAllocation));
  ebpf::AllocaOp allocaOp = cast<ebpf::AllocaOp>(pktAllocation);
  for (auto &block : regions.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      LogicalResult _ = llvm::TypeSwitch<Operation *, LogicalResult>(&op)
                            // ebpf ops.
                            .Case<ebpf::LoadPktPtrOp>([&](auto op) {
                              return replaceLoadPktPtrWithGep(op, allocaOp);
                            })
                            .Default([&](Operation *) { return failure(); });
    }
  }
}

/// @brief Find uses of GetAddrOp and replace storeOp with storeAddrOp
/// @param op
/// @return success if we replace a load address with loadAddr
LogicalResult useStoreAddr(ebpf::GetAddrOp &op) {
  if (op.getResult().use_empty()) {
    return success();
  }
  assert(op.getResult().hasOneUse());
  mlir::Operation *oneUse = op.getResult().use_begin().getUser();
  if (isa<ebpf::StoreOp>(oneUse)) {
    ebpf::StoreOp temp = cast<ebpf::StoreOp>(oneUse);
    auto m_context = temp.getOperation()->getContext();
    auto m_builder = OpBuilder(m_context);
    m_builder.setInsertionPointAfterValue(temp.offset());
    m_builder.create<ebpf::StoreAddrOp>(temp.getLoc(), temp.lhs(),
                                        temp.offset(), temp.rhs());
    oneUse->erase();
    return success();
  }
  return success();
}

/// @brief Store the loaded packet pointer
/// @param funcOp expects to receive a file with xdp_entry inlined into main
void handleStoreAddrForLoadedPktPtr(mlir::Operation &funcOp) {
  auto &regions = funcOp.getRegion(0);
  for (auto &block : regions.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      LogicalResult _ =
          llvm::TypeSwitch<Operation *, LogicalResult>(&op)
              // ebpf ops.
              .Case<ebpf::GetAddrOp>([&](auto op) { return useStoreAddr(op); })
              .Default([&](Operation *) { return failure(); });
    }
  }
}

struct ResolveMemoryPass : public ResolveMemoryBase<ResolveMemoryPass> {
  void runOnOperation() override {
    assert(isa<mlir::ModuleOp>(getOperation()));
    mlir::ModuleOp rootOp = getOperation();
    auto &topBlock = rootOp.body().getBlocks().front();
    assert(topBlock.getOperations().size() == 2 &&
           "there are more than 2 functions");
    assert(isa<mlir::FuncOp>(topBlock.getOperations().front())); // xdp_entry
    assert(isa<mlir::FuncOp>(topBlock.getOperations().back()));  // main
    // remove xdp_entry
    auto &xdpEntryFunc = topBlock.getOperations().front();
    xdpEntryFunc.erase();
    // process main
    auto &funcOp = topBlock.getOperations().back();
    processLoadAddrPattern(funcOp);
    adjustLoadPktPtrOps(funcOp);
    handleStoreAddrForLoadedPktPtr(funcOp);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::ebpf::resolveMemory() {
  return std::make_unique<ResolveMemoryPass>();
}
