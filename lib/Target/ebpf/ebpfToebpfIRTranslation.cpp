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

using namespace mlir;
using namespace mlir::ebpf;

void Deserialize::createJmpOp(Jmp jmp, label_t cur_label) {
  // std::cout << " --> f:" << jmp.target.from;
  // std::cout << ", t: " << jmp.target.to << std::endl;
  assert(jmp.target.from > cur_label.from);
  InstructionSeq prog = m_sections.front();
  const auto &[label, ins, line_info] = prog.at(jmp.target.from);
  // std::cout << "  l: " << label.from << ", j-f:" << jmp.target.from
  //           << std::endl;
  assert(label.from == jmp.target.from);
  buildJmpOp(cur_label.from, jmp);
  return;
}

void Deserialize::createBinaryOp(Bin bin) {
  using Op = Bin::Op;
  Value rhs, lhs, res;
  lhs = m_registers.at(bin.dst.v);
  if (std::holds_alternative<Imm>(bin.v)) {
    rhs = buildConstantOp(std::get<Imm>(bin.v));
  } else {
    rhs = m_registers.at(std::get<Reg>(bin.v).v);
  }
  switch (bin.op) {
  case Op::MOV:
    // std::cout << "move";
    res = buildBinaryOp<ebpf::MoveOp>(lhs, rhs);
    break;
  case Op::MOVSX8:
    // std::cout << "s8";
    res = buildBinaryOp<ebpf::Move8Op>(lhs, rhs);
    break;
  case Op::MOVSX16:
    // std::cout << "s16";
    res = buildBinaryOp<ebpf::Move16Op>(lhs, rhs);
    break;
  case Op::MOVSX32:
    // std::cout << "s32";
    res = buildBinaryOp<ebpf::Move32Op>(lhs, rhs);
    break;
  case Op::ADD:
    // std::cout << "+";
    res = buildBinaryOp<ebpf::AddOp>(lhs, rhs);
    break;
  case Op::SUB:
    // std::cout << "-";
    res = buildBinaryOp<ebpf::SubOp>(lhs, rhs);
    break;
  case Op::MUL:
    // std::cout << "*";
    res = buildBinaryOp<ebpf::MulOp>(lhs, rhs);
    break;
  case Op::UDIV:
    // std::cout << "/";
    res = buildBinaryOp<ebpf::UDivOp>(lhs, rhs);
    break;
  case Op::SDIV:
    // std::cout << "s/";
    res = buildBinaryOp<ebpf::SDivOp>(lhs, rhs);
    break;
  case Op::UMOD:
    // std::cout << "%";
    res = buildBinaryOp<ebpf::UModOp>(lhs, rhs);
    break;
  case Op::SMOD:
    // std::cout << "s%";
    res = buildBinaryOp<ebpf::SModOp>(lhs, rhs);
    break;
  case Op::OR:
    // std::cout << "|";
    res = buildBinaryOp<ebpf::OrOp>(lhs, rhs);
    break;
  case Op::AND:
    // std::cout << "&";
    res = buildBinaryOp<ebpf::AndOp>(lhs, rhs);
    break;
  case Op::LSH:
    // std::cout << "<<";
    res = buildBinaryOp<ebpf::LSHOp>(lhs, rhs);
    break;
  case Op::RSH:
    // std::cout << ">>";
    res = buildBinaryOp<ebpf::RSHOp>(lhs, rhs);
    break;
  case Op::ARSH:
    // std::cout << ">>>";
    res = buildBinaryOp<ebpf::ShiftRAOp>(lhs, rhs);
    break;
  case Op::XOR:
    // std::cout << "^";
    res = buildBinaryOp<ebpf::XOrOp>(lhs, rhs);
    break;
  }
  // std::cout << std::endl;
  m_registers.at(bin.dst.v) = res;
  return;
}

void Deserialize::createMLIR(Instruction ins, label_t cur_label) {
  std::cout << cur_label.from << " ";
  if (std::holds_alternative<Undefined>(ins)) {
    // std::cout << "undefined" << std::endl;
    return;
  } else if (std::holds_alternative<Bin>(ins)) {
    auto binOp = std::get<Bin>(ins);
    // std::cout << "bin: ";
    createBinaryOp(binOp);
    return;
  } else if (std::holds_alternative<Un>(ins)) {
    std::cout << "unary" << std::endl;
    return;
  } else if (std::holds_alternative<LoadMapFd>(ins)) {
    std::cout << "LoadMapFd" << std::endl;
    return;
  } else if (std::holds_alternative<Call>(ins)) {
    std::cout << "Call" << std::endl;
    return;
  } else if (std::holds_alternative<Callx>(ins)) {
    std::cout << "Callx" << std::endl;
    return;
  } else if (std::holds_alternative<Exit>(ins)) {
    std::cout << "Exit" << std::endl;
    return;
  } else if (std::holds_alternative<Jmp>(ins)) {
    auto jmpOp = std::get<Jmp>(ins);
    // std::cout << "Jmp: ";
    createJmpOp(jmpOp, cur_label);
    return;
  } else if (std::holds_alternative<Mem>(ins)) {
    std::cout << "Mem" << std::endl;
    return;
  } else if (std::holds_alternative<Packet>(ins)) {
    std::cout << "Packet" << std::endl;
    return;
  } else if (std::holds_alternative<Assume>(ins)) {
    std::cout << "Assume" << std::endl;
    return;
  } else if (std::holds_alternative<Atomic>(ins)) {
    std::cout << "Atomic" << std::endl;
    return;
  } else if (std::holds_alternative<Assert>(ins)) {
    std::cout << "Assert" << std::endl;
    return;
  } else if (std::holds_alternative<IncrementLoopCounter>(ins)) {
    std::cout << "IncrementLoopCounter" << std::endl;
    return;
  }
  std::cout << "unknown" << std::endl;
}

void Deserialize::buildFunctionBody() {
  // get first section
  InstructionSeq prog = m_sections.front();
  collectBlocks();
  std::cout << prog.size() << " instructions" << std::endl;
  size_t cur_op = 0;
  for (const size_t next : m_startOfNextBlock) {
    assert(m_jumpBlocks.contains(cur_op));
    Block *curBlock = m_jumpBlocks.at(cur_op);
    m_builder.setInsertionPointToEnd(curBlock);
    std::cout << "NEW block at: " << cur_op << std::endl;
    std::cout << "  next: " << next << std::endl;
    // setup registers to match block arguments
    for (size_t i = 0; i < m_ebpfRegisters; ++i) {
      m_registers.at(i) = curBlock->getArgument(i);
    }
    for (; cur_op < next; ++cur_op) {
      const LabeledInstruction &labeled_inst = prog.at(cur_op);
      const auto &[label, ins, _] = labeled_inst;
      createMLIR(ins, label);
    }
    if (curBlock->empty() ||
        !curBlock->back().mightHaveTrait<OpTrait::IsTerminator>()) {
      assert(m_jumpBlocks.contains(next));
      m_builder.setInsertionPointToEnd(curBlock);
      m_builder.create<BranchOp>(m_unknownLoc, m_jumpBlocks.at(next),
                                 m_lastBlock->getArguments());
      std::cout << "/**/ cpp block at: " << next << std::endl;
      m_lastBlock = m_jumpBlocks.at(next);
    }
  }
  if (cur_op < prog.size()) {
    m_builder.setInsertionPointToEnd(m_lastBlock);
    // setup registers to match block arguments
    for (size_t i = 0; i < m_ebpfRegisters; ++i) {
      m_registers.at(i) = m_lastBlock->getArgument(i);
    }
  }
  for (; cur_op < prog.size(); ++cur_op) {
    const LabeledInstruction &labeled_inst = prog.at(cur_op);
    const auto &[label, ins, _] = labeled_inst;
    createMLIR(ins, label);
  }
  // for (const LabeledInstruction& labeled_inst : prog) {
  //   const auto& [label, ins, line_info] = labeled_inst;
  //   createMLIR(ins, label);
  // }
}

void Deserialize::collectBlocks() {
  auto prog = m_sections.front();
  for (const LabeledInstruction &labeled_inst : prog) {
    const auto &[label, ins, line_info] = labeled_inst;
    if (std::holds_alternative<Jmp>(ins)) {
      auto jmp = std::get<Jmp>(ins);
      auto jmpTo = jmp.target.from;
      if (!m_jmpTargets.contains(jmpTo)) {
        incrementBlocks(jmpTo);
      }
      if (jmp.cond.has_value()) {
        // assume that the next instruction is defined
        incrementBlocks(label.from + 1);
      }
    }
  }
  assert(m_numBlocks == m_startOfNextBlock.size());
  std::sort(m_startOfNextBlock.begin(), m_startOfNextBlock.end());
  std::cout << "we need " << m_numBlocks << " blocks" << std::endl;
}

OwningOpRef<FuncOp> Deserialize::buildXDPFunction() {
  auto regType = m_builder.getI64Type();
  std::vector<Type> argTypes(m_ebpfRegisters, regType);
  // create xdp_entry function with two pointer parameters
  OperationState state(m_unknownLoc, FuncOp::getOperationName());
  FuncOp::build(m_builder, state, "xdp_entry",
                FunctionType::get(m_context, {argTypes}, {regType}));
  OwningOpRef<FuncOp> funcOp = cast<FuncOp>(Operation::create(state));
  std::vector<Location> argLocs(m_ebpfRegisters, funcOp->getLoc());
  Region &region = funcOp->getBody();
  OpBuilder::InsertionGuard guard(m_builder);
  auto *body = m_builder.createBlock(&region, {}, {argTypes}, {argLocs});
  m_builder.setInsertionPointToStart(body);
  // book keeping for future blocks
  updateBlocksMap(body, 0);
  m_lastBlock = body;
  // setup registers
  m_registers = std::vector<mlir::Value>(m_ebpfRegisters, nullptr);
  for (size_t i = 0; i < m_ebpfRegisters; ++i) {
    m_registers.at(i) = body->getArgument(i);
  }
  // m_registers.at(REG::R1_ARG) = body->getArguments().front();
  // m_registers.at(REG::R10_STACK_POINTER) = body->getArguments().back();
  // build function body
  buildFunctionBody();
  // add return statement at final block
  m_builder.setInsertionPointToEnd(m_lastBlock);
  m_registers.at(REG::R0_RETURN_VALUE) = m_lastBlock->getArguments().front();
  assert(m_registers.at(REG::R0_RETURN_VALUE) != nullptr);
  m_builder.create<ReturnOp>(m_unknownLoc, m_registers.at(REG::R0_RETURN_VALUE));
  // // make call to init function to initialize latches
  // auto initResults = buildInitFunction(returnTypes);
  // auto opPosition = m_builder.getInsertionPoint();
  // // Create infinite loop that inlines next function
  // std::vector<Location> returnLocs(m_states.size(), funcOp->getLoc());
  // Block *loopBlock =
  //     m_builder.createBlock(body->getParent(), {}, {returnTypes},
  //     {returnLocs});
  // auto nextResults = buildNextFunction(returnTypes, loopBlock);
  // m_builder.create<BranchOp>(m_unknownLoc, loopBlock, nextResults);
  // // add call to branch from original basic block
  // m_builder.setInsertionPoint(body, opPosition);
  // m_builder.create<BranchOp>(m_unknownLoc, loopBlock, initResults);

  return funcOp;
}

bool Deserialize::parseModelIsSuccessful() {
  if (!m_modelFile)
    return false;
  std::vector<raw_program> raw_progs;
  ebpf_platform_t platform = g_ebpf_platform_linux;
  ebpf_verifier_options_t ebpf_verifier_options = ebpf_verifier_default_options;
  raw_progs = read_elf(m_modelFile, std::string(), std::string(),
                       &ebpf_verifier_options, &platform);
  for (const raw_program &raw_prog : raw_progs) {
    // Convert the raw program section to a set of instructions.
    std::variant<InstructionSeq, std::string> prog_or_error =
        unmarshal(raw_prog);
    if (std::holds_alternative<std::string>(prog_or_error)) {
      std::cout << "unmarshaling error at "
                << std::get<std::string>(prog_or_error) << "\n";
      continue;
    }
    auto &prog = std::get<InstructionSeq>(prog_or_error);
    m_sections.push_back(prog);
    print(prog, std::cout, {});
    // // Convert the instruction sequence to a control-flow graph.
    // cfg_t cfg = prepare_cfg(prog, raw_prog.info,
    // !ebpf_verifier_options.no_simplify); print_dot(cfg, std::cout);
  }
  return !m_sections.empty();
}

static OwningOpRef<ModuleOp> deserializeModule(const llvm::MemoryBuffer *input,
                                               MLIRContext *context) {
  context->loadDialect<ebpf::ebpfDialect, StandardOpsDialect>();

  OwningOpRef<ModuleOp> owningModule(ModuleOp::create(FileLineColLoc::get(
      context, input->getBufferIdentifier(), /*line=*/0, /*column=*/0)));

  Deserialize deserialize(context, input->getBufferIdentifier().str());
  if (deserialize.parseModelIsSuccessful()) {
    OwningOpRef<FuncOp> XDPFunc = deserialize.buildXDPFunction();
    if (!XDPFunc)
      return owningModule;

    owningModule->getBody()->push_back(XDPFunc.release());
  }

  return owningModule;
}

namespace mlir {
namespace ebpf {
void registerebpfTranslation() {
  TranslateToMLIRRegistration fromEBPF(
      "import-ebpf", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        // get section name
        assert(sourceMgr.getNumBuffers() == 1 && "expected one buffer");
        return deserializeModule(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context);
      });
}
} // namespace ebpf
} // namespace mlir
