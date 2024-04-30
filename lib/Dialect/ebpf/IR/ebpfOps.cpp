//===- ebpfOps.cpp - ebpf dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/ebpf/IR/ebpf.h"

using namespace mlir;
using namespace mlir::ebpf;

namespace {
//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  return i1Type;
}

//===----------------------------------------------------------------------===//
// Compare Operations
//===----------------------------------------------------------------------===//

template <typename Op> LogicalResult verifyCmpOp(Op op) {
  Type type = op.result().getType();
  if (!type.isSignlessInteger()) {
    return op.emitOpError() << "result must be a signless integer";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

template <typename Op> LogicalResult verifyConstantOp(Op op) {
  auto resultType = op.result().getType();
  auto attributeType = op.valueAttr().getType();
  if (resultType && attributeType &&
      resultType.getIntOrFloatBitWidth() ==
          attributeType.getIntOrFloatBitWidth()) {
    return success();
  }
  return failure();
}

} // namespace

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/ebpf/IR/ebpfOps.cpp.inc"
