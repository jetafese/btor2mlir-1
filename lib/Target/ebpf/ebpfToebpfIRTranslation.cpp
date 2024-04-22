#include "Target/ebpf/ebpfToebpfIRTranslation.h"
#include "Dialect/ebpf/IR/ebpf.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <iostream>
#include <string>
#include <regex>


using std::regex;
using std::regex_match;

#define LABEL R"_((<\w[a-zA-Z_0-9]*>))_"

using namespace mlir;
using namespace mlir::ebpf;

bool Deserialize::parseModelIsSuccessful() {
  if (!m_modelFile) return false;

  std::vector<raw_program> raw_progs;
  ebpf_platform_t platform = g_ebpf_platform_linux;
  raw_progs = read_elf(m_modelFile, std::string(), std::string(), nullptr, &platform);
  for (const raw_program& raw_prog : raw_progs) {
      std::cout << raw_prog.section << " ";
  }
  return true;
}

static OwningOpRef<ModuleOp> deserializeModule(const llvm::MemoryBuffer *input,
                                               MLIRContext *context) {
  context->loadDialect<ebpf::ebpfDialect, StandardOpsDialect>();

  OwningOpRef<ModuleOp> owningModule(ModuleOp::create(FileLineColLoc::get(
      context, input->getBufferIdentifier(), /*line=*/0, /*column=*/0)));

  Deserialize deserialize(context, input->getBufferIdentifier().str());
  if (deserialize.parseModelIsSuccessful()) {
    std::cout << "success" << std::endl;
    // OwningOpRef<FuncOp> mainFunc = deserialize.buildMainFunction();
    // if (!mainFunc)
    //   return owningModule;

    // owningModule->getBody()->push_back(mainFunc.release());
  }

  return owningModule;
}

namespace mlir {
namespace ebpf {
void registerebpfTranslation() {
  TranslateToMLIRRegistration fromBtor(
      "import-ebpf", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        assert(sourceMgr.getNumBuffers() == 1 && "expected one buffer");
        return deserializeModule(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context);
      });
}
} // namespace ebpf
} // namespace mlir
