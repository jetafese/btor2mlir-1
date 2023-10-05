//===- BtorTypes.cpp - Btor dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/Btor.h"
#include "mlir/IR/Builders.h"
#include "Dialect/Btor/IR/BtorTypes.h"

using namespace mlir;
using namespace mlir::btor;

#include "Dialect/Btor/IR/BtorOpsDialect.cpp.inc"

void BtorDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Btor/IR/BtorOpsTypes.cpp.inc"
      >();
}
