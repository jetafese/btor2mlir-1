//===----------------------------------------------------------------------===//
//
// This provides registration calls for ebpf to ebpf IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_EBPF_EBPFTOEBPFIRTRANSLATION_H
#define TARGET_EBPF_EBPFTOEBPFIRTRANSLATION_H

#include "Dialect/ebpf/IR/ebpf.h"
#include "Dialect/ebpf/IR/ebpfTypes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "crab/cfg.hpp"
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

  Deserialize(MLIRContext *context, const std::string &s, bool ssa,
              int sectionNumber)
      : m_context(context), m_builder(OpBuilder(m_context)),
        m_unknownLoc(UnknownLoc::get(m_context)), m_ssa(ssa),
        m_sectionNumber(sectionNumber) {
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
  OwningOpRef<mlir::FuncOp> buildMainFunction(mlir::ModuleOp module);
  void buildMemFunctionBody();
  void buildSSAFunctionBody();
  void buildFunctionBodyFromCFG();

private:
  ///===----------------------------------------------------------------------===//
  /// Parse ebpf file
  ///===----------------------------------------------------------------------===//

  std::ifstream m_modelFile;
  StringAttr m_sourceFile = nullptr;
  const size_t m_ebpfRegisters = 11;
  const size_t m_ebpf_stack = 512;
  const size_t m_xdpParameters = 2;
  const size_t m_xdp_pkt = 4096;
  const std::string m_xdp_entry = "xdp_entry";

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

  std::vector<size_t> m_startOfNextBlock;
  std::vector<mlir::Value> m_registers;
  std::map<size_t, size_t> m_jmpTargets;
  raw_program m_raw_prog;
  InstructionSeq m_section;
  cfg_t m_cfg;
  std::map<int, Block *> m_bbs;
  std::map<int, int> m_nextCondBlock;

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
  bool m_ssa;
  int m_sectionNumber;

  std::vector<Block *> m_blocks;
  Block *m_lastBlock = nullptr;
  std::map<size_t, Block *> m_jumpBlocks;
  std::vector<bool> m_regIsMapElement =
      std::vector<bool>(m_ebpfRegisters, false);

  void setupXDPEntry(mlir::ModuleOp module);
  void setupRegisters(Block *block);
  void createMLIR(Instruction ins, label_t cur_label);
  void createJmpOp(Jmp jmp, label_t cur_label);
  void createBinaryOp(Bin bin);
  void createUnaryOp(Un un);
  void createMemOp(Mem mem);
  void createLoadMapOp(LoadMapFd loadMap);
  void createNDOp(bool isMapLoad);
  void createAtomicOp(Atomic atomic);
  void collectBlocks();
  void createAssertOp();

  void updateBlocksMap(Block *block, size_t firstOp) {
    m_blocks.push_back(block);
    m_jumpBlocks[firstOp] = block;
  }

  void updateBBMap(Block *block, int label) {
    m_bbs[label] = block;
  }

  void setRegister(const uint8_t idx, const mlir::Value &value,
                   bool isMapLoad = false) {
    if (m_ssa) {
      m_registers.at(idx) = value;
      return;
    }
    m_regIsMapElement.at(idx) = isMapLoad;
    auto zero = buildConstantOp(0);
    auto addr = m_registers.at(idx);
    m_builder.create<ebpf::StoreOp>(m_unknownLoc, addr, zero, value);
  }

  mlir::Value getRegister(const uint8_t idx) {
    auto reg = m_registers.at(idx);
    if (m_ssa) {
      return reg;
    }
    auto zero = buildConstantOp(0);
    auto val = m_builder.create<ebpf::LoadOp>(m_unknownLoc, reg, zero);
    return val;
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

  void buildJumpCFG(Jmp jmp, label_t cur_label) {
    OpBuilder::InsertionGuard guard(m_builder);
    assert(m_bbs.contains(cur_label.from));

    Block *curBlock = m_bbs.at(cur_label.from);
    m_builder.setInsertionPointToEnd(curBlock);
    auto opPosition = m_builder.getInsertionPoint();
    // new blocks
    Block *condBlock = nullptr, *toBlock = nullptr;

    int to = jmp.target.from;
    bool isConditional = jmp.cond.has_value();

    if (isConditional) {
      assert(m_nextCondBlock.contains(cur_label.from));
      auto condLabel = m_nextCondBlock.at(cur_label.from);
      if (!m_bbs.contains(condLabel)) {
        // create the block for the next operation
        condBlock = m_builder.createBlock(curBlock->getParent(), {});
        updateBBMap(condBlock, condLabel);
      } else {
        condBlock = m_bbs.at(condLabel);
      }
    }
    if (!m_bbs.contains(to)) {
      // create the to block
      toBlock = m_builder.createBlock(curBlock->getParent(), {});
      updateBBMap(toBlock, to);
    } else {
      toBlock = m_bbs.at(to);
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
        rhs = getRegister(rhsId);
      }
      auto cmpOp = m_builder.create<CmpOp>(m_unknownLoc, getPred(cond.op),
                                           getRegister(lhsId), rhs);
      m_builder.create<CondBranchOp>(m_unknownLoc, cmpOp, toBlock, condBlock);
    } else {
      m_builder.create<BranchOp>(m_unknownLoc, toBlock);
    }
    return;
  }

  mlir::Value buildConstantOp(Imm imm) {
    auto type = m_builder.getI64Type();
    auto immVal = m_builder.create<ebpf::ConstantOp>(
        m_unknownLoc, type, m_builder.getIntegerAttr(type, imm.v));
    std::cerr << "--created imm const: " << imm.v << std::endl;
    return immVal;
  }

  mlir::Value buildConstantOp(int64_t value) {
    auto type = m_builder.getI64Type();
    auto immVal = m_builder.create<ebpf::ConstantOp>(
        m_unknownLoc, type, m_builder.getIntegerAttr(type, value));
    std::cerr << "--created val const: " << value << std::endl;
    return immVal;
  }

  template <typename ebpfOp>
  mlir::Value buildBinaryOp(const Value &lhs, const Value &rhs) {
    auto res = m_builder.create<ebpfOp>(m_unknownLoc, lhs, rhs);
    return res;
  }

  template <typename ebpfOp> void buildLoadOp(const Value &offset, Mem mem) {
    auto src = getRegister(mem.access.basereg.v);
    Value res;
    if (m_regIsMapElement.at(mem.access.basereg.v)) {
      // TODO: use width specific move operations
      res = src;
    } else {
      res = m_builder.create<ebpfOp>(m_unknownLoc, src, offset);
    }
    setRegister(std::get<Reg>(mem.value).v, res);
  }

  template <typename ebpfOp> void buildStoreOp(const Value &offset, Mem mem) {
    Value writeVal;
    auto baseReg = mem.access.basereg.v;
    if (std::holds_alternative<Imm>(mem.value)) {
      writeVal = buildConstantOp(std::get<Imm>(mem.value));
    } else {
      writeVal = getRegister(std::get<Reg>(mem.value).v);
    }
    // m_regIsMapElement.at(mem.access.basereg.v) = false;
    if (m_regIsMapElement.at(baseReg)) {
      setRegister(baseReg, writeVal);
      return;
    }
    auto base = getRegister(baseReg);
    m_builder.create<ebpfOp>(m_unknownLoc, base, offset, writeVal);
  }

  template <typename ebpfOp> mlir::Value buildUnaryOp(const Value &op) {
    auto res = m_builder.create<ebpfOp>(m_unknownLoc, op);
    return res;
  }
};

/// Register the ebpf translation
void registerebpfTranslation();
void registerebpfMemTranslation();

} // namespace ebpf
} // namespace mlir

#endif // TARGET_EBPF_EBPFTOEBPFIRTRANSLATION_H
