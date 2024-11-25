//===- cex.h - Counter Example Harness for SeaHorn ----------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CEX_H
#define CEX_H

extern "C" {

/*extern void __VERIFIER_error(void);*/
/*extern void __VERIFIER_assert(bool);*/
void __VERIFIER_error(void) {}
void __VERIFIER_assert(int cond) {
  if (!(cond)) {
    __VERIFIER_error();
  }
}

#define sassert(X)                                                             \
  (void)((__VERIFIER_assert(X), (X)) || (__VERIFIER_error(), 0))
}

#endif
