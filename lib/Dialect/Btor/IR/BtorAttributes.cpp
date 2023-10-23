//===- BtorTypes.cpp - Btor dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/Btor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "Dialect/Btor/IR/BtorTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "Dialect/Btor/IR/BtorAttributes.h"



using namespace mlir;
using namespace mlir::btor;

#define GET_ATTRDEF_CLASSES
#include "Dialect/Btor/IR/BtorAttributes.cpp.inc"

void BtorDialect::registerAttrs() {
   addTypes<
#define GET_ATTRDEF_LIST
#include "Dialect/Btor/IR/BtorAttributes.cpp.inc"
      >();
}
