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
  // TODO: clean up the special case handling, perhaps use setRegister
  if (mem.is_load) {
    auto baseReg = mem.access.basereg.v;
    if (m_reg_types.at(baseReg) == REG_TYPE::CTX) {
      auto loadOffset = mem.access.offset;
      auto ctx = m_raw_prog.info.type.context_descriptor;
      if (loadOffset == ctx->data || loadOffset == ctx->end) {
        m_reg_types.at(std::get<Reg>(mem.value).v) = REG_TYPE::PKT;
        auto res = m_builder.create<ebpf::LoadPktPtrOp>(m_unknownLoc, offset);
        setRegister(std::get<Reg>(mem.value).v, res);
        return;
      }
    }
    if (m_reg_types.at(baseReg) == REG_TYPE::PKT) {
      m_reg_types.at(std::get<Reg>(mem.value).v) = REG_TYPE::PKT;
      auto res = m_builder.create<ebpf::LoadPktPtrOp>(m_unknownLoc, offset);
      setRegister(std::get<Reg>(mem.value).v, res);
      return;
    }
  }

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
  Value mapSize;
  auto dst = loadMap.dst.v;
  assert(m_mapDescriptorFdToSize.contains(loadMap.mapfd));
  mapSize = buildConstantOp(m_mapDescriptorFdToSize.at(loadMap.mapfd));
  setRegister(dst, mapSize);
}

void Deserialize::createNDOp(bool isMapLoad = false) {
  Value res = m_builder.create<NDOp>(m_unknownLoc, m_builder.getI64Type());
  setRegister(0, res, isMapLoad);
}

void Deserialize::createMapLookupOp(Call lookup) {
  auto mapFd = getRegister(lookup.singles.front().reg.v, true);
  auto key = getRegister(lookup.singles.back().reg.v);
  Value res = m_builder.create<ebpf::MapLookupOp>(m_unknownLoc, mapFd, key);
  setRegister(0, res, true);
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

/// @brief Creates mlir instruction for every ebpf instruction
/// @tparam ebpf Instruction, basis block label
/// @return none (builds instruction in mlir basic block)
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
    if (callOp.is_map_lookup) {
      std::cerr << "Map Lookup" << std::endl;
      createMapLookupOp(callOp);
      return;
    }
    std::size_t found;
    for (const auto &fname : m_functionNames) {
      found = callOp.name.find(fname);
      if (found != std::string::npos) {
        std::cerr << "Call " << fname << std::endl;
        createNDOp();
        return;
      }
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
    createNDOp();
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

/// @brief Builds corresponding function from ebpf CFG
/// @tparam Starting (entry) basic block
/// @return none (ensures that function has been built in mlir)
void Deserialize::buildFunctionBodyFromCFG(Block *body) {
  cfg_t m_cfg = get_cfg(m_sectionIns);
  // collect bbs in a vector
  std::vector<label_t> bbLabels;
  for (auto const &[this_label, _] : m_cfg) {
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
  m_reg_types = std::vector<REG_TYPE>(m_ebpfRegisters, REG_TYPE::NUM);
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
  m_reg_types.at(REG::R1_ARG) = REG_TYPE::CTX;
  m_reg_types.at(REG::R10_STACK_POINTER) = REG_TYPE::STACK;
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
  /* initialize ctx so that data begin/end point to pkt begin/end */
  auto ctx_size = m_raw_prog.info.type.context_descriptor->size;
  assert(ctx_size > 0);
  Value ctx = buildConstantOp(ctx_size);
  auto data_begin = m_raw_prog.info.type.context_descriptor->data;
  auto data_end = m_raw_prog.info.type.context_descriptor->end;
  assert(((data_begin == -1) && (data_end == -1)) || (data_begin < data_end));
  auto ctxPtr = m_builder.create<ebpf::AllocaOp>(m_unknownLoc, ctx);
  m_builder.create<ebpf::MemHavocOp>(m_unknownLoc, ctxPtr, ctx);
  m_builder.create<ebpf::Store32Op>(
      m_unknownLoc, ctxPtr, buildConstantOp(data_begin == -1 ? 0 : data_begin),
      buildConstantOp(0));
  m_builder.create<ebpf::Store32Op>(
      m_unknownLoc, ctxPtr, buildConstantOp(data_end == -1 ? 4 : data_end),
      pkt);
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
  try {
    raw_progs = read_elf(m_modelFile, m_sourceFile.str(), m_section,
                         &ebpf_verifier_options, &platform);
  } catch (std::runtime_error &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return false;
  }
  if (m_section.empty() || m_function.empty()) {
    std::cerr << "please specify a program with: -section <string>";
    std::cerr << " -function <string>\n";
    std::cerr << "available programs (section, function):\n";
    raw_progs = read_elf(m_modelFile, std::string(), std::string(),
                         &ebpf_verifier_options, &platform);
    for (const raw_program &raw_prog : raw_progs) {
      std::cerr << raw_prog.section_name << ", " << raw_prog.function_name
                << "\n";
    }
    return false;
  }
  bool functionFound = false;
  for (raw_program &raw_prog : raw_progs) {
      if (raw_prog.function_name == m_function) {
          m_raw_prog = raw_prog;
          functionFound = true;
      }
  }
  if (!functionFound) {
    std::cerr << "function is not found in the given section\n";
    return false;
  }
  for (const EbpfMapDescriptor &desc : m_raw_prog.info.map_descriptors) {
    m_mapDescriptorFdToSize[desc.original_fd] = desc.value_size;
  }
  // Convert the raw program section to a set of instructions.
  std::variant<InstructionSeq, std::string> prog_or_error =
      unmarshal(m_raw_prog);
  if (std::holds_alternative<std::string>(prog_or_error)) {
    std::cerr << "unmarshaling error at "
              << std::get<std::string>(prog_or_error) << "\n";
    return false;
  }
  m_sectionIns = std::get<InstructionSeq>(prog_or_error);
  print(m_sectionIns, std::cerr, {});
  return m_sectionIns.size() > 0;
}

static OwningOpRef<ModuleOp> deserializeModule(const llvm::MemoryBuffer *input,
                                               MLIRContext *context,
                                               std::string section,
                                               std::string function) {
  context->loadDialect<ebpf::ebpfDialect, StandardOpsDialect>();

  OwningOpRef<ModuleOp> owningModule(ModuleOp::create(FileLineColLoc::get(
      context, input->getBufferIdentifier(), /*line=*/0, /*column=*/0)));

  Deserialize deserialize(context, input->getBufferIdentifier().str(), section,
                          function);
  if (!deserialize.parseModelIsSuccessful()) {
    exit(1);
  }

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

  return owningModule;
}

//===----------------------------------------------------------------------===//
// Translation CommandLine Options
//===----------------------------------------------------------------------===//
static llvm::cl::opt<std::string> sectionOpt("section", llvm::cl::init(""),
                                             llvm::cl::desc("section"));
static llvm::cl::opt<std::string> funcionOpt("function", llvm::cl::init(""),
                                             llvm::cl::desc("function"));

namespace mlir {
namespace ebpf {
void registerebpfMemTranslation() {
  TranslateToMLIRRegistration fromEBPF(
      "import-ebpf-mem", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        // get section name
        assert(sourceMgr.getNumBuffers() == 1 && "expected one buffer");
        return deserializeModule(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context,
            sectionOpt.getValue(), funcionOpt.getValue());
      });
}
} // namespace ebpf
} // namespace mlir
