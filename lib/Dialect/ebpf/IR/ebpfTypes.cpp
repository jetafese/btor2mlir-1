//===- ebpfTypes.cpp - ebpf dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/ebpf/IR/ebpfTypes.h"
#include "Dialect/ebpf/IR/ebpf.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ebpf;

#define GET_TYPEDEF_CLASSES
#include "Dialect/ebpf/IR/ebpfOpsTypes.cpp.inc"

void ebpfDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/ebpf/IR/ebpfOpsTypes.cpp.inc"
      >();
}