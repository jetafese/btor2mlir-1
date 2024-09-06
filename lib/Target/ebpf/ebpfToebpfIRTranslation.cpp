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
  buildJumpCFG(jmp, cur_label);
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
  Value rhs, res;
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
    res = buildBinaryOp<ebpf::AddOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::SUB:
    res = buildBinaryOp<ebpf::SubOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::MUL:
    res = buildBinaryOp<ebpf::MulOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::UDIV:
    res = buildBinaryOp<ebpf::UDivOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::SDIV:
    res = buildBinaryOp<ebpf::SDivOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::UMOD:
    res = buildBinaryOp<ebpf::UModOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::SMOD:
    res = buildBinaryOp<ebpf::SModOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::OR:
    res = buildBinaryOp<ebpf::OrOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::AND:
    res = buildBinaryOp<ebpf::AndOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::LSH:
    res = buildBinaryOp<ebpf::LSHOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::RSH:
    res = buildBinaryOp<ebpf::RSHOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::ARSH:
    res = buildBinaryOp<ebpf::ShiftRAOp>(getRegister(bin.dst.v), rhs);
    break;
  case Op::XOR:
    res = buildBinaryOp<ebpf::XOrOp>(getRegister(bin.dst.v), rhs);
    break;
  }
  setRegister(bin.dst.v, res);
  return;
}

void Deserialize::createMemOp(Mem mem) {
  auto offset = buildConstantOp(mem.access.offset);
  switch (mem.access.width) {
  case 1:
    if (mem.is_load) {
      buildLoadOp<ebpf::Load8Op>(offset, mem);
    } else {
      buildStoreOp<ebpf::Store8Op>(offset, mem);
    }
    break;
  case 2:
    if (mem.is_load) {
      buildLoadOp<ebpf::Load16Op>(offset, mem);
    } else {
      buildStoreOp<ebpf::Store16Op>(offset, mem);
    }
    break;
  case 4:
    if (mem.is_load) {
      buildLoadOp<ebpf::Load32Op>(offset, mem);
    } else {
      buildStoreOp<ebpf::Store32Op>(offset, mem);
    }
    break;
  case 8:
    if (mem.is_load) {
      buildLoadOp<ebpf::LoadOp>(offset, mem);
    } else {
      buildStoreOp<ebpf::StoreOp>(offset, mem);
    }
    break;
  }
}

void Deserialize::createLoadMapOp(LoadMapFd loadMap) {
  Value res, map;
  auto dst = loadMap.dst.v;
  map = buildConstantOp(loadMap.mapfd);
  res = buildBinaryOp<ebpf::LoadMapOp>(getRegister(dst), map);
  setRegister(dst, res);
}

void Deserialize::createNDOp(bool isMapLoad = false) {
  Value res = m_builder.create<NDOp>(m_unknownLoc, m_builder.getI64Type());
  setRegister(0, res, isMapLoad);
}

void Deserialize::createAssertOp() {
  auto type = m_builder.getI1Type();
  auto falseVal = m_builder.create<ebpf::ConstantOp>(
      m_unknownLoc, type, m_builder.getIntegerAttr(type, 0));
  m_builder.create<ebpf::AssertOp>(m_unknownLoc, falseVal);
}

void Deserialize::createAtomicOp(Atomic atomicOp) {
  Value lhs, base, offset, rhs, res;
  base = getRegister(atomicOp.access.basereg.v);
  offset = buildConstantOp(atomicOp.access.offset);
  rhs = getRegister(atomicOp.valreg.v);
  switch (atomicOp.access.width) {
  case 1:
    lhs = m_builder.create<ebpf::Load8Op>(m_unknownLoc, base, offset);
    break;
  case 2:
    lhs = m_builder.create<ebpf::Load16Op>(m_unknownLoc, base, offset);
    break;
  case 4:
    lhs = m_builder.create<ebpf::Load32Op>(m_unknownLoc, base, offset);
    break;
  case 8:
    lhs = m_builder.create<ebpf::LoadOp>(m_unknownLoc, base, offset);
    break;
  }
  using Op = Atomic::Op;
  switch (atomicOp.op) {
  case Op::ADD:
    std::cerr << "add" << std::endl;
    res = buildBinaryOp<ebpf::AddOp>(lhs, rhs);
    break;
  case Op::AND:
    std::cerr << "and" << std::endl;
    res = buildBinaryOp<ebpf::AndOp>(lhs, rhs);
    break;
  case Op::OR:
    std::cerr << "or" << std::endl;
    res = buildBinaryOp<ebpf::OrOp>(lhs, rhs);
    break;
  case Op::XOR:
    std::cerr << "xor" << std::endl;
    res = buildBinaryOp<ebpf::XOrOp>(lhs, rhs);
    break;
  case Op::XCHG:
    std::cerr << "xchg" << std::endl;
    assert(false);
    break;
  case Op::CMPXCHG:
    std::cerr << "cmpxchg" << std::endl;
    assert(false);
    break;
  }
  setRegister(atomicOp.access.basereg.v, res);
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
      std::cerr << "Call get_prandom" << std::endl;
      createAssertOp();
      return;
    }
    if (callOp.is_map_lookup) {
      std::cerr << "Map Lookup" << std::endl;
      createNDOp(true);
      return;
    }
    found = callOp.name.find("redirect_map");
    if (found != std::string::npos) {
      std::cerr << "Call redirect_map" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("perf_event_output");
    if (found != std::string::npos) {
      std::cerr << "Call perf_event_output" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("get_hash_recalc");
    if (found != std::string::npos) {
      std::cerr << "Call get_hash_recalc" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("skb_");
    if (found != std::string::npos) {
      std::cerr << "Call skb_*" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("csum_diff");
    if (found != std::string::npos) {
      std::cerr << "Call csum_diff" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("l4_csum_replace");
    if (found != std::string::npos) {
      std::cerr << "Call l4_csum_replace" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("tail_call");
    if (found != std::string::npos) {
      std::cerr << "Call tail_call" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("map_update_elem");
    if (found != std::string::npos) {
      std::cerr << "Call map_update_elem" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("jiffies64");
    if (found != std::string::npos) {
      std::cerr << "Call jiffies64" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("map_delete_elem");
    if (found != std::string::npos) {
      std::cerr << "Call map_delete_elem" << std::endl;
      createNDOp();
      return;
    }
    found = callOp.name.find("redirect");
    if (found != std::string::npos) {
      std::cerr << "Call redirect" << std::endl;
      createNDOp();
      return;
    }
    std::cerr << "Other Call: " << callOp.name << std::endl;
    assert(false);
    return;
  } else if (std::holds_alternative<Callx>(ins)) {
    std::cerr << "Callx" << std::endl;
    assert(false);
    return;
  } else if (std::holds_alternative<Exit>(ins)) {
    // std::cerr << "Exit" << std::endl;
    if (!m_bbs.contains(-2)) {
      auto jmpExit = Jmp{
          .target = label_t{-2},
      };
      createJmpOp(jmpExit, cur_label);
    }
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
    std::cerr << "Atomic: ";
    auto atomicOp = std::get<Atomic>(ins);
    createAtomicOp(atomicOp);
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

void Deserialize::buildFunctionBodyFromCFG(Block *body) {
  cfg_t m_cfg = get_cfg(m_section);
  // collect bbs in a vector
  std::vector<label_t> bbLabels;
  for (auto const &[this_label, _] : m_cfg) {
    std::cerr << this_label.from << ", ";
    bbLabels.push_back(this_label);
  }
  assert(bbLabels.size() == m_cfg.size() && "expect similar size");
  // build mlir
  size_t bbIter = 0;
  bool terminatorSet = false;
  for (auto const &[this_label, bb] : m_cfg) {
    if (this_label.from == -1) {
      // entrance block
      updateBBMap(body, this_label.from);
    }
    terminatorSet = false;
    assert(m_bbs.contains(this_label.from));
    m_builder.setInsertionPointToEnd(m_bbs.at(this_label.from));
    if (this_label.from == -2) {
      // exit block
      Value ret = getRegister(REG::R0_RETURN_VALUE);
      assert(ret != nullptr);
      m_builder.create<ReturnOp>(m_unknownLoc, ret);
      continue;
    }

    for (const auto &ins : bb) {
      assert(!terminatorSet && "jump instruction before end of bb");
      if (std::holds_alternative<Jmp>(ins)) {
        auto jmpOp = std::get<Jmp>(ins);
        if (jmpOp.cond.has_value()) {
          /* store next block when conditional*/
          assert(bbIter + 1 < bbLabels.size());
          m_nextCondBlock[this_label.from] = bbLabels.at(bbIter + 1).from;
        }
        createMLIR(ins, this_label);
        terminatorSet = true;
      } else if (std::holds_alternative<Exit>(ins)) {
        createMLIR(ins, this_label);
        terminatorSet = true;
      } else {
        createMLIR(ins, this_label);
      }
    }
    if (!terminatorSet) {
      auto jmpFallthrough = Jmp{
          .target = bbLabels.at(bbIter + 1),
      };
      createMLIR(jmpFallthrough, bbLabels.at(bbIter));
    }
    bbIter++;
  }
}

void Deserialize::setupRegisters(Block *body) {
  m_registers = std::vector<mlir::Value>(m_ebpfRegisters, nullptr);
  Value allocaSize = buildConstantOp(8);
  for (size_t i = 0; i < m_ebpfRegisters; ++i) {
    Value reg = m_builder.create<ebpf::AllocaOp>(m_unknownLoc, allocaSize);
    m_registers.at(i) = reg;
  }
  /* r1 and r10 are pointers to ctx and stack respectively*/
  Value zero_offset = buildConstantOp(0);
  m_builder.create<ebpf::StoreAddrOp>(m_unknownLoc, m_registers.at(REG::R1_ARG),
                                      zero_offset, body->getArgument(0));
  m_builder.create<ebpf::StoreAddrOp>(m_unknownLoc,
                                      m_registers.at(REG::R10_STACK_POINTER),
                                      zero_offset, body->getArgument(1));
}

OwningOpRef<FuncOp> Deserialize::buildXDPFunction() {
  auto regType = m_builder.getI64Type();
  std::vector<Type> argTypes = std::vector<Type>(m_xdpParameters, regType);
  // create xdp_entry function with parameters
  OperationState state(m_unknownLoc, FuncOp::getOperationName());
  FuncOp::build(m_builder, state, m_xdp_entry,
                FunctionType::get(m_context, {argTypes}, {regType}));
  OwningOpRef<FuncOp> funcOp = cast<FuncOp>(Operation::create(state));
  std::vector<Location> argLocs =
      std::vector<Location>(m_xdpParameters, funcOp->getLoc());
  Region &region = funcOp->getBody();
  OpBuilder::InsertionGuard guard(m_builder);
  auto *body = m_builder.createBlock(&region, {}, {argTypes}, {argLocs});
  m_builder.setInsertionPointToStart(body);
  // setup registers
  setupRegisters(body);
  // build function body
  buildFunctionBodyFromCFG(body);
  return funcOp;
}

void Deserialize::setupXDPEntry(ModuleOp module) {
  /* setup packet, ctx */
  Value pkt = buildConstantOp(m_xdp_pkt);
  auto pktPtr = m_builder.create<ebpf::AllocaOp>(m_unknownLoc, pkt);
  m_builder.create<ebpf::MemHavocOp>(m_unknownLoc, pktPtr, pkt);
  auto endPktPtr = m_builder.create<ebpf::GetAddrOp>(
      m_unknownLoc, pktPtr, buildConstantOp(m_xdp_pkt - 1));
  /* initialize ctx so that data begin/end point to pkt begin/end */
  auto ctx_size = m_raw_prog.info.type.context_descriptor->size;
  assert(ctx_size > 0);
  Value ctx = buildConstantOp(ctx_size);
  auto data_begin = m_raw_prog.info.type.context_descriptor->data;
  auto data_end = m_raw_prog.info.type.context_descriptor->end;
  assert(((data_begin == -1) && (data_end == -1)) || (data_begin < data_end));
  auto ctxPtr = m_builder.create<ebpf::AllocaOp>(m_unknownLoc, ctx);
  m_builder.create<ebpf::MemHavocOp>(m_unknownLoc, ctxPtr, ctx);
  m_builder.create<ebpf::StoreAddrOp>(
      m_unknownLoc, ctxPtr, buildConstantOp(data_begin == -1 ? 0 : data_begin),
      pktPtr);
  m_builder.create<ebpf::StoreAddrOp>(
      m_unknownLoc, ctxPtr, buildConstantOp(data_end == -1 ? 4 : data_begin),
      endPktPtr);
  /* initialzie stack; stack ptr should point to end of stack*/
  Value stack = buildConstantOp(m_ebpf_stack);
  auto stackBlock = m_builder.create<ebpf::AllocaOp>(m_unknownLoc, stack);
  m_builder.create<ebpf::MemHavocOp>(m_unknownLoc, stackBlock, stack);
  Value endOfStack = buildConstantOp(m_ebpf_stack - 8);
  auto stackPtr =
      m_builder.create<ebpf::GetAddrOp>(m_unknownLoc, stackBlock, endOfStack);
  /* call xdp_entry */
  auto xdpEntryFunc = module.lookupSymbol<FuncOp>(m_xdp_entry);
  assert(xdpEntryFunc);
  auto callXDP = m_builder
                     .create<CallOp>(m_unknownLoc, xdpEntryFunc,
                                     ValueRange({ctxPtr, stackPtr}))
                     .getResult(0);
  m_builder.create<ReturnOp>(m_unknownLoc, callXDP);
}

OwningOpRef<FuncOp> Deserialize::buildMainFunction(ModuleOp module) {
  OperationState state(m_unknownLoc, FuncOp::getOperationName());
  FuncOp::build(m_builder, state, "main",
                FunctionType::get(m_context, {}, {m_builder.getI64Type()}));
  OwningOpRef<FuncOp> funcOp = cast<FuncOp>(Operation::create(state));
  Region &region = funcOp->getBody();
  OpBuilder::InsertionGuard guard(m_builder);
  auto *body = m_builder.createBlock(&region, {}, {}, {});
  m_builder.setInsertionPointToStart(body);
  /* setup ctx, stack, packet*/
  setupXDPEntry(module);
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
  size_t sect = 0;
  std::vector<std::string> desired_sections =
      std::vector<std::string>(raw_progs.size());
  if (raw_progs.size() == 1) {
    m_sectionNumber = 0;
  } else if ((m_sectionNumber < 0) || (m_sectionNumber >= raw_progs.size())) {
    std::cerr << "please specify a section with: -section <int>\n";
    std::cerr << "available sections:\n";
    for (const raw_program &raw_prog : raw_progs) {
      desired_sections.at(sect) = raw_prog.section;
      std::cerr << sect++ << ": " << raw_prog.section << "\n";
    }
    return false;
  }
  m_raw_prog = raw_progs.at(m_sectionNumber);
  // Convert the raw program section to a set of instructions.
  std::variant<InstructionSeq, std::string> prog_or_error =
      unmarshal(m_raw_prog);
  if (std::holds_alternative<std::string>(prog_or_error)) {
    std::cerr << "unmarshaling error at "
              << std::get<std::string>(prog_or_error) << "\n";
    return false;
  }
  m_section = std::get<InstructionSeq>(prog_or_error);
  print(m_section, std::cerr, {});
  return m_section.size() > 0;
}

static OwningOpRef<ModuleOp> deserializeModule(const llvm::MemoryBuffer *input,
                                               MLIRContext *context,
                                               int sectionNumber) {
  context->loadDialect<ebpf::ebpfDialect, StandardOpsDialect>();

  OwningOpRef<ModuleOp> owningModule(ModuleOp::create(FileLineColLoc::get(
      context, input->getBufferIdentifier(), /*line=*/0, /*column=*/0)));

  Deserialize deserialize(context, input->getBufferIdentifier().str(),
                          sectionNumber);
  if (deserialize.parseModelIsSuccessful()) {
    OwningOpRef<FuncOp> XDPFunc = deserialize.buildXDPFunction();
    if (!XDPFunc) {
      return owningModule;
    }
    owningModule->getBody()->push_back(XDPFunc.release());

    OwningOpRef<FuncOp> mainFunc =
        deserialize.buildMainFunction(owningModule.get());
    if (!mainFunc) {
      return owningModule;
    }
    owningModule->getBody()->push_back(mainFunc.release());
  }

  return owningModule;
}

//===----------------------------------------------------------------------===//
// Translation CommandLine Options
//===----------------------------------------------------------------------===//
static llvm::cl::opt<int> sectionOpt("section", llvm::cl::init(-1),
                                     llvm::cl::desc("section number"));

namespace mlir {
namespace ebpf {
void registerebpfMemTranslation() {
  TranslateToMLIRRegistration fromEBPF(
      "import-ebpf-mem", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        // get section name
        assert(sourceMgr.getNumBuffers() == 1 && "expected one buffer");
        return deserializeModule(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context,
            sectionOpt.getValue());
      });
}
} // namespace ebpf
} // namespace mlir
