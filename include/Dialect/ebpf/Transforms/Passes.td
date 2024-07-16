//===-- Passes.td - StandardOps pass definition file -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_DIALECT_TRANSFORMS_PASSES
#define EBPF_DIALECT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ResolveMemory : Pass<"resolve-mem", "ModuleOp"> {
  let summary = "Resolve ebpf memory in mlir module";
  let constructor = "mlir::ebpf::resolveMemory()";
  let dependentDialects = [];
}

#endif // EBPF_DIALECT_TRANSFORMS_PASSES