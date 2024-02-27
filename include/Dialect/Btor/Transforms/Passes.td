//===-- Passes.td - StandardOps pass definition file -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_DIALECT_TRANSFORMS_PASSES
#define BTOR_DIALECT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def BtorLiveness : Pass<"btor-liveness", "ModuleOp"> {
  let summary = "Compute liveness for btor ops";
  let constructor = "mlir::btor::computeBtorLiveness()";
  let dependentDialects = [];
}

def ResolveCasts : Pass<"resolve-casts", "ModuleOp"> {
  let summary = "Resolve casts in mlir module";
  let constructor = "mlir::btor::resolveCasts()";
  let dependentDialects = [];
}

#endif // BTOR_DIALECT_TRANSFORMS_PASSES