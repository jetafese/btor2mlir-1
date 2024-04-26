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
  if (jmp.cond.has_value()) {
    auto op = jmp.cond.value().op;
    using Op = Condition::Op;
    switch (op) {
    case Op::EQ:
      std::cout << "==";
      break;
    case Op::NE:
      std::cout << "!=";
      break;
    case Op::SET:
      std::cout << "&==";
      break;
    case Op::NSET:
      std::cout << "&!=";
      break; // not in ebpf
    case Op::LT:
      std::cout << "<";
      break; // TODO: os << "u<";
    case Op::LE:
      std::cout << "<=";
      break; // TODO: os << "u<=";
    case Op::GT:
      std::cout << ">";
      break; // TODO: os << "u>";
    case Op::GE:
      std::cout << ">=";
      break; // TODO: os << "u>=";
    case Op::SLE:
      std::cout << "s<=";
      break;
    case Op::SLT:
      std::cout << "s<";
      break;
    case Op::SGT:
      std::cout << "s>";
      break;
    case Op::SGE:
      std::cout << "s>=";
      break;
    }
  }
  std::cout << " --> f:" << jmp.target.from;
  std::cout << ", t: " << jmp.target.to << std::endl;
  assert(jmp.target.from > cur_label.from);
  InstructionSeq prog = m_sections.front();
  const auto &[label, ins, line_info] = prog.at(jmp.target.from);
  std::cout << "  l: " << label.from << ", j-f:" << jmp.target.from
            << std::endl;
  assert(label.from == jmp.target.from);
  std::cout << "    j to: ";
  createMLIR(ins, label);
  buildJmpOp(cur_label.from, jmp.target.from, jmp.cond.has_value());
  return;
}

void Deserialize::createBinaryOp(Bin bin) {
  using Op = Bin::Op;
  switch (bin.op) {
  case Op::MOV:
    std::cout << "move";
    break;
  case Op::MOVSX8:
    std::cout << "s8";
    break;
  case Op::MOVSX16:
    std::cout << "s16";
    break;
  case Op::MOVSX32:
    std::cout << "s32";
    break;
  case Op::ADD:
    std::cout << "+";
    break;
  case Op::SUB:
    std::cout << "-";
    break;
  case Op::MUL:
    std::cout << "*";
    break;
  case Op::UDIV:
    std::cout << "/";
    break;
  case Op::SDIV:
    std::cout << "s/";
    break;
  case Op::UMOD:
    std::cout << "%";
    break;
  case Op::SMOD:
    std::cout << "s%";
    break;
  case Op::OR:
    std::cout << "|";
    break;
  case Op::AND:
    std::cout << "&";
    break;
  case Op::LSH:
    std::cout << "<<";
    break;
  case Op::RSH:
    std::cout << ">>";
    break;
  case Op::ARSH:
    std::cout << ">>>";
    break;
  case Op::XOR:
    std::cout << "^";
    break;
  }
  std::cout << std::endl;
  return;
}

void Deserialize::createMLIR(Instruction ins, label_t cur_label) {
  std::cout << cur_label.from << " ";
  if (std::holds_alternative<Undefined>(ins)) {
    std::cout << "undefined" << std::endl;
    return;
  } else if (std::holds_alternative<Bin>(ins)) {
    auto binOp = std::get<Bin>(ins);
    std::cout << "bin: ";
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
    std::cout << "Jmp: ";
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

void Deserialize::buildFunctionBody(const std::vector<Value> &registers) {
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
  auto regType = m_builder.getIntegerType(64, false);
  std::vector<Type> argTypes(m_xdpParameters, regType);
  // create xdp_entry function with two pointer parameters
  OperationState state(m_unknownLoc, FuncOp::getOperationName());
  FuncOp::build(m_builder, state, "xdp_entry",
                FunctionType::get(m_context, {argTypes}, {regType}));
  OwningOpRef<FuncOp> funcOp = cast<FuncOp>(Operation::create(state));
  std::vector<Location> argLocs(m_xdpParameters, funcOp->getLoc());
  Region &region = funcOp->getBody();
  OpBuilder::InsertionGuard guard(m_builder);
  auto *body = m_builder.createBlock(&region, {}, {argTypes}, {argLocs});
  m_builder.setInsertionPointToStart(body);
  // book keeping for future blocks
  updateBlocksMap(body, 0);
  m_lastBlock = body;
  // setup registers
  std::vector<Value> registers(m_ebpfRegisters, nullptr);
  registers.at(REG::R1_ARG) = body->getArguments().front();
  registers.at(REG::R10_STACK_POINTER) = body->getArguments().back();
  // build function body
  buildFunctionBody(registers);
  // add return statement at final block
  m_builder.setInsertionPointToEnd(m_lastBlock);
  registers.at(REG::R0_RETURN_VALUE) = m_lastBlock->getArguments().front();
  assert(registers.at(REG::R0_RETURN_VALUE) != nullptr);
  m_builder.create<ReturnOp>(m_unknownLoc, registers.at(REG::R0_RETURN_VALUE));
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
