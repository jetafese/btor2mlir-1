//===- ebpfDialect.cpp - ebpf dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/ebpf/IR/ebpf.h"

#include "Dialect/ebpf/IR/ebpfOpsDialect.cpp.inc"

// Pull in all enum type definitions and utility function declarations.
#include "Dialect/ebpf/IR/ebpfOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::ebpf;

//===----------------------------------------------------------------------===//
// ebpf dialect.
//===----------------------------------------------------------------------===//

void ebpfDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/ebpf/IR/ebpfOps.cpp.inc"
      >();
}
