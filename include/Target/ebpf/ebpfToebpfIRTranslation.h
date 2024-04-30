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

#include <fstream>
#include <map>
#include <utility>
#include <vector>

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

  OwningOpRef<mlir::FuncOp> buildXDPFunction();
  void buildFunctionBody();

private:
  ///===----------------------------------------------------------------------===//
  /// Parse ebpf file
  ///===----------------------------------------------------------------------===//

  std::ifstream m_modelFile;
  StringAttr m_sourceFile = nullptr;
  const size_t m_ebpfRegisters = 11;
  const size_t m_xdpParameters = 2;

  enum REG : size_t {
    R0_RETURN_VALUE = 0,
    R1_ARG = 1,
    R2_ARG = 2,
    R3_ARG = 3,
    R4_ARG = 4,
    R5_ARG = 5,
    R6 = 6,
    R7 = 7,
    R8 = 8,
    R9 = 9,
    R10_STACK_POINTER = 10
  };

  std::vector<InstructionSeq> m_sections;
  std::vector<size_t> m_startOfNextBlock;
  std::vector<mlir::Value> m_registers;
  std::map<size_t, size_t> m_jmpTargets;

  size_t m_numBlocks = 0;
  void incrementBlocks(size_t jmpTo) {
    if (setInsWithLabel(jmpTo)) {
      m_startOfNextBlock.push_back(jmpTo);
      m_numBlocks++;
    }
  }

  size_t getInsByLabel(const size_t label) { return m_jmpTargets.at(label); }

  bool setInsWithLabel(const size_t label) {
    if (m_jmpTargets.contains(label))
      return false;
    m_jmpTargets[label] = label;
    return true;
  }
  ///===----------------------------------------------------------------------===//
  /// Create MLIR module
  ///===----------------------------------------------------------------------===//

  MLIRContext *m_context;
  OpBuilder m_builder;
  Location m_unknownLoc;

  std::vector<Block *> m_blocks;
  Block *m_lastBlock = nullptr;
  std::map<size_t, Block *> m_jumpBlocks;

  void createMLIR(Instruction ins, label_t cur_label);
  void createJmpOp(Jmp jmp, label_t cur_label);
  void createBinaryOp(Bin bin);
  void collectBlocks();

  void updateBlocksMap(Block *block, size_t firstOp) {
    m_blocks.push_back(block);
    m_jumpBlocks[firstOp] = block;
  }

  void buildJmpOp(size_t from, size_t to, bool isConditional) {
    OpBuilder::InsertionGuard guard(m_builder);
    m_builder.setInsertionPointToEnd(m_lastBlock);
    auto opPosition = m_builder.getInsertionPoint();
    Block *condBlock = nullptr, *toBlock = nullptr;
    std::vector<Location> returnLocs(2, m_unknownLoc);
    Block *curBlock = m_lastBlock;

    assert(to > from && "backjumps not implemented yet");

    if (isConditional) {
      if (!m_jumpBlocks.contains(from + 1)) {
        // create the another block for the next operation
        condBlock =
            m_builder.createBlock(curBlock->getParent(), {},
                                  {curBlock->getArgumentTypes()}, {returnLocs});
        updateBlocksMap(condBlock, from + 1);
        std::cout << "*** create condBlock at: " << from + 1 << std::endl;
      } else {
        condBlock = m_jumpBlocks.at(from + 1);
      }
    }
    if (!m_jumpBlocks.contains(to)) {
      // create the to block
      toBlock =
          m_builder.createBlock(curBlock->getParent(), {},
                                {curBlock->getArgumentTypes()}, {returnLocs});
      updateBlocksMap(toBlock, to);
      std::cout << "*** create toBlock at: " << to << std::endl;
    } else {
      toBlock = m_jumpBlocks.at(to);
    }
    m_builder.setInsertionPoint(curBlock, opPosition);
    m_builder.create<BranchOp>(m_unknownLoc, condBlock,
                               curBlock->getArguments());
    std::cout << "/**/ end block at: " << from << std::endl;
    m_lastBlock = condBlock;
    return;
  }
};

/// Register the ebpf translation
void registerebpfTranslation();

} // namespace ebpf
} // namespace mlir

#endif // TARGET_EBPF_EBPFTOEBPFIRTRANSLATION_H
