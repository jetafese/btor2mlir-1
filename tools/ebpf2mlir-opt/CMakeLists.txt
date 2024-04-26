get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRebpf
        )
add_llvm_executable(ebpf2mlir-opt ebpf2mlir-opt.cpp)

llvm_update_compile_flags(ebpf2mlir-opt)
target_link_libraries(ebpf2mlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(ebpf2mlir-opt)
