#ifndef BTOR_TYPES
#define BTOR_TYPES

include "BtorBase.td"
//include "mlir/IR/AttrTypeBase.td"

class Btor_BaseType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Btor_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Btor_BitVec : Btor_BaseType<"BitVec", "bv"> {
  let summary = "bit vector behaving similar to unsinged int";
  let description = [{
    Bit vectors have designated length.
  }];

  let parameters = (ins "unsigned":$length);

  let assemblyFormat = "`<` $length `>`";

  /// Indicate that our type will add additional verification to the parameters.
  let genVerifyDecl = 1;
}

def Btor_Array : Btor_BaseType<"Array", "array"> {
  let summary = "array that's a pair of bit vectors";
  let description = [{
    Two bit vectors acting as a Btor array
  }];

  let parameters = (ins "BitVecType":$a, "BitVecType":$b);

  let assemblyFormat = "`<` $a `,` $b `>`";

  /// Indicate that our type will add additional verification to the parameters.
  let genVerifyDecl = 1;
}

#endif // BTOR_TYPES