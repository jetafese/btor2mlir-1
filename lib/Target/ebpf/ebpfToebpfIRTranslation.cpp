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

#define MINUS1_32 4294967295
#define MINUS1_16 65535
#define MINUS1_8 255

void Deserialize::createJmpOp(Jmp jmp, label_t cur_label) {
  // std::cerr << " --> f:" << jmp.target.from;
  // std::cerr << ", t: " << jmp.target.to << std::endl;
  assert(jmp.target.from > cur_label.from);
  InstructionSeq prog = m_sections.front();
  const auto &[label, ins, line_info] = prog.at(jmp.target.from);
  // std::cerr << "  l: " << label.from << ", j-f:" << jmp.target.from
  //           << std::endl;
  assert(label.from == jmp.target.from);
  buildJmpOp(cur_label.from, jmp);
  return;
}

void Deserialize::createUnaryOp(Un un) {
  using Op = Un::Op;
  Value rhs, res;
  rhs = getRegister(un.dst.v);
  switch (un.op) {
  case Op::BE16:
    res = buildUnaryOp<ebpf::BE16>(rhs);
    break;
  case Op::BE32:
    res = buildUnaryOp<ebpf::BE32>(rhs);
    break;
  case Op::BE64:
    res = buildUnaryOp<ebpf::BE64>(rhs);
    break;
  case Op::LE16:
    res = buildUnaryOp<ebpf::LE16>(rhs);
    break;
  case Op::LE32:
    res = buildUnaryOp<ebpf::LE32>(rhs);
    break;
  case Op::LE64:
    res = buildUnaryOp<ebpf::LE64>(rhs);
    break;
  case Op::SWAP16:
    res = buildUnaryOp<ebpf::SWAP16>(rhs);
    break;
  case Op::SWAP32:
    res = buildUnaryOp<ebpf::SWAP32>(rhs);
    break;
  case Op::SWAP64:
    res = buildUnaryOp<ebpf::SWAP64>(rhs);
    break;
  case Op::NEG:
    res = buildUnaryOp<ebpf::NegOp>(rhs);
    break;
  }
  setRegister(un.dst.v, res);
}

void Deserialize::createBinaryOp(Bin bin) {
  using Op = Bin::Op;
  Value rhs, lhs, res;
  lhs = getRegister(bin.dst.v);
  if (std::holds_alternative<Imm>(bin.v)) {
    rhs = buildConstantOp(std::get<Imm>(bin.v));
  } else {
    rhs = getRegister(std::get<Reg>(bin.v).v);
  }
  switch (bin.op) {
  case Op::MOV:
    res = rhs;
    break;
  case Op::MOVSX8:
    /* mask to isolate the 8bits*/
    res = m_builder.create<ebpf::AndOp>(m_unknownLoc, rhs,
                                        buildConstantOp(MINUS1_8));
    break;
  case Op::MOVSX16:
    /* mask to isolate the 16bits*/
    res = m_builder.create<ebpf::AndOp>(m_unknownLoc, rhs,
                                        buildConstantOp(MINUS1_16));
    break;
  case Op::MOVSX32:
    /* mask to isolate the 32bits*/
    res = m_builder.create<ebpf::AndOp>(m_unknownLoc, rhs,
                                        buildConstantOp(MINUS1_32));
    break;
  case Op::ADD:
    res = buildBinaryOp<ebpf::AddOp>(lhs, rhs);
    break;
  case Op::SUB:
    res = buildBinaryOp<ebpf::SubOp>(lhs, rhs);
    break;
  case Op::MUL:
    res = buildBinaryOp<ebpf::MulOp>(lhs, rhs);
    break;
  case Op::UDIV:
    res = buildBinaryOp<ebpf::UDivOp>(lhs, rhs);
    break;
  case Op::SDIV:
    res = buildBinaryOp<ebpf::SDivOp>(lhs, rhs);
    break;
  case Op::UMOD:
    res = buildBinaryOp<ebpf::UModOp>(lhs, rhs);
    break;
  case Op::SMOD:
    res = buildBinaryOp<ebpf::SModOp>(lhs, rhs);
    break;
  case Op::OR:
    res = buildBinaryOp<ebpf::OrOp>(lhs, rhs);
    break;
  case Op::AND:
    res = buildBinaryOp<ebpf::AndOp>(lhs, rhs);
    break;
  case Op::LSH:
    res = buildBinaryOp<ebpf::LSHOp>(lhs, rhs);
    break;
  case Op::RSH:
    res = buildBinaryOp<ebpf::RSHOp>(lhs, rhs);
    break;
  case Op::ARSH:
    res = buildBinaryOp<ebpf::ShiftRAOp>(lhs, rhs);
    break;
  case Op::XOR:
    res = buildBinaryOp<ebpf::XOrOp>(lhs, rhs);
    break;
  }
  setRegister(bin.dst.v, res);
  return;
}

void Deserialize::createMemOp(Mem mem) {
  Value res;
  auto offset = buildConstantOp(mem.access.offset);
  switch (mem.access.width) {
  case 1:
    res = (mem.is_load)
              ? buildBinaryOp<ebpf::Load8Op>(getRegister(mem.access.basereg.v),
                                             offset)
              : buildStoreOp<ebpf::Store8Op>(getRegister(mem.access.basereg.v),
                                             offset, mem);
    break;
  case 2:
    res = (mem.is_load)
              ? buildBinaryOp<ebpf::Load16Op>(getRegister(mem.access.basereg.v),
                                              offset)
              : buildStoreOp<ebpf::Store16Op>(getRegister(mem.access.basereg.v),
                                              offset, mem);
    break;
  case 4:
    res = (mem.is_load)
              ? buildBinaryOp<ebpf::Load32Op>(getRegister(mem.access.basereg.v),
                                              offset)
              : buildStoreOp<ebpf::Store32Op>(getRegister(mem.access.basereg.v),
                                              offset, mem);
    break;
  case 8:
    res = (mem.is_load)
              ? buildBinaryOp<ebpf::LoadOp>(getRegister(mem.access.basereg.v),
                                            offset)
              : buildStoreOp<ebpf::StoreOp>(getRegister(mem.access.basereg.v),
                                            offset, mem);
    break;
  }
  if (mem.is_load) {
    setRegister(std::get<Reg>(mem.value).v, res);
  }
}

void Deserialize::createLoadMapOp(LoadMapFd loadMap) {
  Value res, map;
  auto dst = loadMap.dst.v;
  map = buildConstantOp(loadMap.mapfd);
  res = buildBinaryOp<ebpf::LoadMapOp>(getRegister(dst), map);
  setRegister(dst, res);
}

void Deserialize::createNDOp() {
  Value res = m_builder.create<NDOp>(m_unknownLoc, m_builder.getI64Type());
  setRegister(0, res);
}

void Deserialize::createAssertOp() {
  auto type = m_builder.getI1Type();
  auto falseVal = m_builder.create<ebpf::ConstantOp>(
      m_unknownLoc, type, m_builder.getIntegerAttr(type, 0));
  m_builder.create<ebpf::AssertOp>(m_unknownLoc, falseVal);
}

void Deserialize::createMLIR(Instruction ins, label_t cur_label) {
  std::cerr << cur_label.from << " ";
  if (std::holds_alternative<Undefined>(ins)) {
    // std::cerr << "undefined" << std::endl;
    return;
  } else if (std::holds_alternative<Bin>(ins)) {
    auto binOp = std::get<Bin>(ins);
    // std::cerr << "bin: ";
    createBinaryOp(binOp);
    return;
  } else if (std::holds_alternative<Un>(ins)) {
    auto unOp = std::get<Un>(ins);
    // std::cerr << "unary" << std::endl;
    createUnaryOp(unOp);
    return;
  } else if (std::holds_alternative<LoadMapFd>(ins)) {
    auto mapOp = std::get<LoadMapFd>(ins);
    std::cerr << "LoadMapFd" << std::endl;
    createLoadMapOp(mapOp);
    return;
  } else if (std::holds_alternative<Call>(ins)) {
    auto callOp = std::get<Call>(ins);
    std::cerr << "-- call: " << callOp.func + 0 << std::endl;
    std::size_t found = callOp.name.find("get_prandom");
    if (found != std::string::npos) {
      createAssertOp();
      return;
    }

    createNDOp();
    if (callOp.is_map_lookup) {
      std::cerr << "Map Lookup" << std::endl;
      return;
    }
    std::cerr << "Other Call" << std::endl;
    // assert(false);
    return;
  } else if (std::holds_alternative<Callx>(ins)) {
    std::cerr << "Callx" << std::endl;
    createNDOp();
    return;
  } else if (std::holds_alternative<Exit>(ins)) {
    // std::cerr << "Exit" << std::endl;
    return;
  } else if (std::holds_alternative<Jmp>(ins)) {
    auto jmpOp = std::get<Jmp>(ins);
    // std::cerr << "Jmp: ";
    createJmpOp(jmpOp, cur_label);
    return;
  } else if (std::holds_alternative<Mem>(ins)) {
    auto memOp = std::get<Mem>(ins);
    // std::cerr << "Mem" << std::endl;
    createMemOp(memOp);
    return;
  } else if (std::holds_alternative<Packet>(ins)) {
    std::cerr << "Packet" << std::endl;
    assert(false);
    return;
  } else if (std::holds_alternative<Assume>(ins)) {
    std::cerr << "Assume" << std::endl;
    assert(false);
    return;
  } else if (std::holds_alternative<Atomic>(ins)) {
    std::cerr << "Atomic" << std::endl;
    assert(false);
    return;
  } else if (std::holds_alternative<Assert>(ins)) {
    std::cerr << "Assert" << std::endl;
    assert(false);
    return;
  } else if (std::holds_alternative<IncrementLoopCounter>(ins)) {
    std::cerr << "IncrementLoopCounter" << std::endl;
    assert(false);
    return;
  }
  assert(false && "unknown");
}

void Deserialize::buildSSAFunctionBody() {
  // get first section
  InstructionSeq prog = m_sections.front();
  collectBlocks();
  std::cerr << prog.size() << " instructions" << std::endl;
  size_t cur_op = 0;
  for (const size_t next : m_startOfNextBlock) {
    assert(m_jumpBlocks.contains(cur_op));
    Block *curBlock = m_jumpBlocks.at(cur_op);
    m_builder.setInsertionPointToEnd(curBlock);
    std::cerr << "NEW block at: " << cur_op << std::endl;
    std::cerr << "  next: " << next << std::endl;
    // setup registers to match block arguments
    for (size_t i = 0; i < m_ebpfRegisters; ++i) {
      setRegister(i, curBlock->getArgument(i));
      // m_registers.at(i) = curBlock->getArgument(i);
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
                                 m_registers);
      std::cerr << "/**/ cpp block at: " << next << std::endl;
      m_lastBlock = m_jumpBlocks.at(next);
    }
  }
  if (cur_op < prog.size()) {
    m_builder.setInsertionPointToEnd(m_lastBlock);
    // setup registers to match block arguments
    for (size_t i = 0; i < m_ebpfRegisters; ++i) {
      setRegister(i, m_lastBlock->getArgument(i));
      // m_registers.at(i) = m_lastBlock->getArgument(i);
    }
  }
  for (; cur_op < prog.size(); ++cur_op) {
    const LabeledInstruction &labeled_inst = prog.at(cur_op);
    const auto &[label, ins, _] = labeled_inst;
    createMLIR(ins, label);
  }
  m_builder.setInsertionPointToEnd(m_lastBlock);
}

void Deserialize::buildMemFunctionBody() {
  // get first section
  InstructionSeq prog = m_sections.front();
  collectBlocks();
  std::cerr << prog.size() << " instructions" << std::endl;
  size_t cur_op = 0;
  for (const size_t next : m_startOfNextBlock) {
    assert(m_jumpBlocks.contains(cur_op));
    Block *curBlock = m_jumpBlocks.at(cur_op);
    m_builder.setInsertionPointToEnd(curBlock);
    std::cerr << "NEW block at: " << cur_op << std::endl;
    std::cerr << "  next: " << next << std::endl;
    for (; cur_op < next; ++cur_op) {
      const LabeledInstruction &labeled_inst = prog.at(cur_op);
      const auto &[label, ins, _] = labeled_inst;
      createMLIR(ins, label);
    }
    if (curBlock->empty() ||
        !curBlock->back().mightHaveTrait<OpTrait::IsTerminator>()) {
      assert(m_jumpBlocks.contains(next));
      m_builder.setInsertionPointToEnd(curBlock);
      m_builder.create<BranchOp>(m_unknownLoc, m_jumpBlocks.at(next));
      std::cerr << "/**/ cpp block at: " << next << std::endl;
      m_lastBlock = m_jumpBlocks.at(next);
    }
  }
  if (cur_op < prog.size()) {
    m_builder.setInsertionPointToEnd(m_lastBlock);
  }
  for (; cur_op < prog.size(); ++cur_op) {
    const LabeledInstruction &labeled_inst = prog.at(cur_op);
    const auto &[label, ins, _] = labeled_inst;
    createMLIR(ins, label);
  }
  m_builder.setInsertionPointToEnd(m_lastBlock);
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
  std::cerr << "we need " << m_numBlocks << " blocks" << std::endl;
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
  Value zero_offset = buildConstantOp(0);
  for (size_t i = 0; i < m_ebpfRegisters; ++i) {
    if (m_ssa) {
      setRegister(i, body->getArgument(i));
      // m_registers.at(i) = body->getArgument(i);
    } else {
      Value reg = m_builder.create<ebpf::AllocaOp>(m_unknownLoc,
                                                   m_builder.getI64Type());
      // setRegister(i, reg);
      m_registers.at(i) = reg;
      m_builder.create<ebpf::StoreOp>(m_unknownLoc, reg, zero_offset,
                                      body->getArgument(i));
    }
  }

  // build function body
  if (m_ssa) {
    buildSSAFunctionBody();
    setRegister(REG::R0_RETURN_VALUE, m_lastBlock->getArguments().front());
    // m_registers.at(REG::R0_RETURN_VALUE) =
    // m_lastBlock->getArguments().front();
  } else {
    buildMemFunctionBody();
  }
  assert(getRegister(REG::R0_RETURN_VALUE) != nullptr);
  m_builder.create<ReturnOp>(m_unknownLoc, getRegister(REG::R0_RETURN_VALUE));
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
      std::cerr << "unmarshaling error at "
                << std::get<std::string>(prog_or_error) << "\n";
      continue;
    }
    auto &prog = std::get<InstructionSeq>(prog_or_error);
    m_sections.push_back(prog);
    print(prog, std::cerr, {});
    // // Convert the instruction sequence to a control-flow graph.
    // cfg_t cfg = prepare_cfg(prog, raw_prog.info,
    // !ebpf_verifier_options.no_simplify); print_dot(cfg, std::cerr);
  }
  return !m_sections.empty();
}

static OwningOpRef<ModuleOp> deserializeModule(const llvm::MemoryBuffer *input,
                                               MLIRContext *context, bool ssa) {
  context->loadDialect<ebpf::ebpfDialect, StandardOpsDialect>();

  OwningOpRef<ModuleOp> owningModule(ModuleOp::create(FileLineColLoc::get(
      context, input->getBufferIdentifier(), /*line=*/0, /*column=*/0)));

  Deserialize deserialize(context, input->getBufferIdentifier().str(), ssa);
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
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context,
            true);
      });
}

void registerebpfMemTranslation() {
  TranslateToMLIRRegistration fromEBPF(
      "import-ebpf-mem", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        // get section name
        assert(sourceMgr.getNumBuffers() == 1 && "expected one buffer");
        return deserializeModule(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context,
            false);
      });
}
} // namespace ebpf
} // namespace mlir
