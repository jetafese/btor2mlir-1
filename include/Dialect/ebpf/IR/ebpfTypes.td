#ifndef EBPF_TYPES
#define EBPF_TYPES

include "ebpfBase.td"

class ebpf_BaseType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ebpf_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ebpf_mem : ebpf_BaseType<"Memory", "mem"> {
  let summary = "ebpf memory";
  let description = [{A ebpf memory parametrized by its size}];

  let parameters = (ins "unsigned" : $size);

  let assemblyFormat = "`<` $size `>`";

  let genVerifyDecl = 0;
}
#endif // EBPF_TYPES
