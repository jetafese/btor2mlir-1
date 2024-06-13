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
  // const size_t m_xdpParameters = 2;

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
  void createUnaryOp(Un un);
  void createMemOp(Mem mem);
  void createLoadMapOp(LoadMapFd loadMap);
  void collectBlocks();

  void updateBlocksMap(Block *block, size_t firstOp) {
    m_blocks.push_back(block);
    m_jumpBlocks[firstOp] = block;
  }

  ebpf::ebpfPredicate getPred(Condition::Op op) {
    using Op = Condition::Op;
    switch (op) {
    case Op::EQ:
      return ebpf::ebpfPredicate::eq;
    case Op::NE:
      return ebpf::ebpfPredicate::ne;
    case Op::SET:
      return ebpf::ebpfPredicate::set;
    case Op::NSET:
      assert(false);
      break; // not in ebpf
    case Op::LT:
      return ebpf::ebpfPredicate::ult;
    case Op::LE:
      return ebpf::ebpfPredicate::ule;
    case Op::GT:
      return ebpf::ebpfPredicate::ugt;
    case Op::GE:
      return ebpf::ebpfPredicate::uge;
    case Op::SLE:
      return ebpf::ebpfPredicate::sle;
    case Op::SLT:
      return ebpf::ebpfPredicate::slt;
    case Op::SGT:
      return ebpf::ebpfPredicate::sgt;
    case Op::SGE:
      return ebpf::ebpfPredicate::sge;
    }
  }

  void buildJmpOp(size_t from, Jmp jmp) {
    OpBuilder::InsertionGuard guard(m_builder);
    m_builder.setInsertionPointToEnd(m_lastBlock);
    auto opPosition = m_builder.getInsertionPoint();
    // new blocks
    Block *condBlock = nullptr, *toBlock = nullptr;
    std::vector<Location> returnLocs(m_ebpfRegisters, m_unknownLoc);
    Block *curBlock = m_lastBlock;

    size_t to = jmp.target.from;
    bool isConditional = jmp.cond.has_value();
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
    // create branch operation for original block
    m_builder.setInsertionPoint(curBlock, opPosition);
    if (isConditional) {
      auto cond = jmp.cond.value();
      auto lhsId = cond.left.v;
      Value rhs;
      if (std::holds_alternative<Imm>(cond.right)) {
        // set rhs to the immediate value
        rhs = buildConstantOp(std::get<Imm>(cond.right));
      } else {
        auto rhsId = std::get<Reg>(cond.right).v;
        rhs = m_registers.at(rhsId);
      }
      auto cmpOp = m_builder.create<CmpOp>(m_unknownLoc, getPred(cond.op),
                                           m_registers.at(lhsId), rhs);
      m_builder.create<CondBranchOp>(m_unknownLoc, cmpOp, toBlock, m_registers,
                                     condBlock, m_registers);
    } else {
      m_builder.create<BranchOp>(m_unknownLoc, condBlock, m_registers);
    }
    std::cout << "/**/ end block at: " << from << std::endl;
    m_lastBlock = condBlock;
    return;
  }

  mlir::Value buildConstantOp(Imm imm) {
    auto type = m_builder.getI64Type();
    auto immVal = m_builder.create<ebpf::ConstantOp>(
        m_unknownLoc, type, m_builder.getIntegerAttr(type, imm.v));
    std::cout << "--created imm const: " << imm.v << std::endl;
    return immVal;
  }

  mlir::Value buildConstantOp(int32_t value) {
    auto type = m_builder.getI64Type();
    auto immVal = m_builder.create<ebpf::ConstantOp>(
        m_unknownLoc, type, m_builder.getIntegerAttr(type, value));
    std::cout << "--created val const: " << value << std::endl;
    return immVal;
  }

  template <typename ebpfOp>
  mlir::Value buildBinaryOp(const Value &lhs, const Value &rhs) {
    auto res = m_builder.create<ebpfOp>(m_unknownLoc, lhs, rhs);
    return res;
  }

  template <typename ebpfOp>
  mlir::Value buildStoreOp(const Value &base, const Value &offset, Mem mem) {
    Value writeVal;
    if (std::holds_alternative<Imm>(mem.value)) {
      writeVal = buildConstantOp(std::get<Imm>(mem.value));
    } else {
      writeVal = m_registers.at(std::get<Reg>(mem.value).v);
    }
    m_builder.create<ebpfOp>(m_unknownLoc, base, offset, writeVal);
    return writeVal;
  }

  template <typename ebpfOp>
  mlir::Value buildUnaryOp(const Value &op) {
    auto res = m_builder.create<ebpfOp>(m_unknownLoc, op);
    return res;
  }
};

/// Register the ebpf translation
void registerebpfTranslation();

} // namespace ebpf
} // namespace mlir

#endif // TARGET_EBPF_EBPFTOEBPFIRTRANSLATION_H
