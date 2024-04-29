//===- ebpfOps.cpp - ebpf dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/ebpf/IR/ebpf.h"

// #define GET_OP_CLASSES
// #include "Dialect/ebpf/IR/ebpfOps.cpp.inc"

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
//   auto i1Type = btor::BitVecType::get(type.getContext(), 1);
//   auto i1Type = Base::get(type.getContext(), 1);
  return type;
}

//===----------------------------------------------------------------------===//
// Compare Operations
//===----------------------------------------------------------------------===//

template <typename Op> LogicalResult verifyCmpOp(Op op) {
//   unsigned resultLength = getBVType(op.result().getType()).getWidth();
//   if (resultLength != 1) {
//     return op.emitOpError()
//            << "result must be a signless integer instead got length of "
//            << resultLength;
//   }
    Type type = op.result().getType();
    if (!type.isSignlessInteger())
    {
        return op.emitOpError()
        << "result must be a signless integer";
    }
  return success();
}

}//namespace

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/ebpf/IR/ebpfOps.cpp.inc"

// //===----------------------------------------------------------------------===//
// // TableGen'd enum attribute definitions
// //===----------------------------------------------------------------------===//

// #include "Dialect/ebpf/IR/ebpfOpsEnums.cpp.inc"
