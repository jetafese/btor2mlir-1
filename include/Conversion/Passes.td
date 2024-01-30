#ifndef BTOR_CONVERSION_PASSES
#define BTOR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// BtorToArithmetic
//===----------------------------------------------------------------------===//

def ConvertBtorToArithmetic : Pass<"convert-btor-to-arith", "ModuleOp"> {
  let summary = "Convert Btor Dialect to Arithmetic dialect";
  // clang-format off
  let description = [{
    Convert btor operations into the Arithmetic dialect operations.

    If other operations are present and their results are required by the Arithmetic 
    dialect operations, the pass will fail.  Any Arithmetic operations or types
    already present in the IR will be kept as is.
  }];
  // clang-format on
  let constructor = "btor::createConvertBtorToArithmeticPass()";
  let dependentDialects = ["arith::ArithmeticDialect"];
}

//===----------------------------------------------------------------------===//
// BtorToLLVM
//===----------------------------------------------------------------------===//

def ConvertBtorToLLVM : Pass<"convert-btor-to-llvm", "ModuleOp"> {
  let summary = "Convert Btor Dialect to LLVM dialect";
  // clang-format off
  let description = [{
    Convert btor operations into LLVM instructions
  }];
  // clang-format on
  let constructor = "mlir::btor::createLowerToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// BtorNDToLLVM
//===----------------------------------------------------------------------===//

def ConvertBtorNDToLLVM : Pass<"convert-btornd-to-llvm", "ModuleOp"> {
  let summary = "Convert Btor Dialect nd Operations to LLVM dialect";
  // clang-format off
  let description = [{
    Convert nd btor operations into LLVM instructions
  }];
  // clang-format on
  let constructor = "mlir::btor::createLowerBtorNDToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// BtorToVector
//===----------------------------------------------------------------------===//

def ConvertBtorToVector : Pass<"convert-btor-to-vector", "ModuleOp"> {
  let summary = "Convert Btor Dialect to Vector dialect";
  // clang-format off
  let description = [{
    Convert btor array operations into vector instructions
  }];
  // clang-format on
  let constructor = "mlir::btor::createLowerToVectorPass()";
  let dependentDialects = ["vector::VectorDialect"];
}

//===----------------------------------------------------------------------===//
// BtorToMemref
//===----------------------------------------------------------------------===//

def ConvertBtorToMemref : Pass<"convert-btor-to-memref", "ModuleOp"> {
  let summary = "Convert Btor Dialect to Memref dialect";
  // clang-format off
  let description = [{
    Convert btor array operations into memref instructions
  }];
  // clang-format on
  let constructor = "mlir::btor::createLowerToMemrefPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

#endif // BTOR_CONVERSION_PASSES
