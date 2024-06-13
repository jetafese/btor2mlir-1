add_mlir_conversion_library(MLIRebpfToLLVM
    ebpfToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Conversion/ebpfToLLVM

    DEPENDS
    BTORConversionPassIncGen
    intrinsics_gen
    
    LINK_COMPONENTS
    Core
    
    LINK_LIBS PUBLIC
    MLIRLLVMCommonConversion
    MLIRLLVMIR
    MLIRTransforms
    )