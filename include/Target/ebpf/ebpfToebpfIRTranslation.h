//===----------------------------------------------------------------------===//
//
// This provides registration calls for ebpf to ebpf IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_EBPF_EBPFTOEBPFIRTRANSLATION_H
#define TARGET_EBPF_EBPFTOEBPFIRTRANSLATION_H

#include "Dialect/ebpf/IR/ebpf.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

#include <map>
#include <utility>
#include <vector>
#include <fstream>

#include "ebpf_verifier.hpp"

namespace mlir {
class MLIRContext;
class ModuleOp;

namespace ebpf {

/// Deserializes the given ebpf module and creates a MLIR ModuleOp
/// in the given `context`.

class Deserialize {

public:
  ///===----------------------------------------------------------------------===//
  /// Constructors and Destructors
  ///===----------------------------------------------------------------------===//

  Deserialize(MLIRContext *context, const std::string &s)
      : m_context(context), m_builder(OpBuilder(m_context)),
        m_unknownLoc(UnknownLoc::get(m_context)) {
    m_modelFile.open(s.c_str());
    m_sourceFile = m_builder.getStringAttr(s);
  }

  ~Deserialize() {}

  ///===----------------------------------------------------------------------===//
  /// Parse ebpf2 file
  ///===----------------------------------------------------------------------===//

  bool parseModelIsSuccessful();

  ///===----------------------------------------------------------------------===//
  /// Create MLIR module
  ///===----------------------------------------------------------------------===//

private:
  ///===----------------------------------------------------------------------===//
  /// Parse ebpf file
  ///===----------------------------------------------------------------------===//

  std::ifstream m_modelFile;
  StringAttr m_sourceFile = nullptr;

  ///===----------------------------------------------------------------------===//
  /// Create MLIR module
  ///===----------------------------------------------------------------------===//

  MLIRContext *m_context;
  OpBuilder m_builder;
  Location m_unknownLoc;

};

/// Register the ebpf translation
void registerebpfTranslation();

} // namespace ebpf
} // namespace mlir

#endif // TARGET_EBPF_EBPFTOEBPFIRTRANSLATION_H
