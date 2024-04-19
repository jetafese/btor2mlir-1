//===- ebpfOps.td - ebpf dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_OPS
#define EBPF_OPS

include "ebpfBase.td"

//===----------------------------------------------------------------------===//
// Base ebpf operation definition.
//===----------------------------------------------------------------------===//

class ebpf_Op<string mnemonic, list<Trait> traits = []> :
        Op<ebpf_Dialect, mnemonic, traits>;

#endif // EBPF_OPS
