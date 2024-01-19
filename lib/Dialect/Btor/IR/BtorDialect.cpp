//===- BtorDialect.cpp - Btor dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/Btor.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::btor;

#include "Dialect/Btor/IR/BtorOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// BtorDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
/// This class defines the interface for handling inlining with btor ops
struct BtorInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  ~BtorInlinerInterface() override = default;

  /// All call operations within BtorDialect can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    // Return true here when inlining into std func operation
    auto *op = dest->getParentOp();
    return isa<mlir::FuncOp>(op);
  }

  /// Returns true if the given operation 'op', that is registered to this
  /// dialect, can be inlined into the region 'dest' that is attached to an
  /// operation registered to the current dialect.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }

  /// BtorDialect terminator operations don't really need any special handing.
  void handleTerminator(Operation *op, Block *newDest) const final {}
};
} // namespace

//===----------------------------------------------------------------------===//
// Btor dialect.
//===----------------------------------------------------------------------===//

void BtorDialect::initialize() {
  registerTypes();
  registerAttrs();
  addOperations<
#define GET_OP_LIST
#include "Dialect/Btor/IR/BtorOps.cpp.inc"
      >();
  addInterfaces<BtorInlinerInterface>();
}
