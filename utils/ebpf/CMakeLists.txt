cmake_minimum_required(VERSION 3.16)
project(fuzz_project LANGUAGES CXX)

# Require the user to specify the compiler and llvm-link paths
if(NOT DEFINED CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "CMAKE_CXX_COMPILER is not defined. Please specify it with -DCMAKE_CXX_COMPILER=<path> when running cmake.")
endif()

if(NOT DEFINED LLVM_LINK)
    message(FATAL_ERROR "LLVM_LINK is not defined. Please specify it with -DLLVM_LINK=<path> when running cmake.")
endif()

# Compilation flags
set(CXXFLAGS -emit-llvm -S)
set(FUZZ_FLAGS -fsanitize=fuzzer,address -g -O1)

# Source files
set(SRC_FILES test.cpp fuzz_helpers.cpp)
set(LL_FILES test.ll fuzz_helpers.ll)
set(LINKED_LL final_linked_file.ll)
set(EXECUTABLE fuzz_executable)

# Add custom commands for each .cpp -> .ll
foreach(src_file IN LISTS SRC_FILES)
    get_filename_component(src_name ${src_file} NAME_WE)
    add_custom_command(
        OUTPUT ${src_name}.ll
        COMMAND ${CMAKE_CXX_COMPILER} ${CXXFLAGS} ${CMAKE_SOURCE_DIR}/${src_file} -o ${CMAKE_BINARY_DIR}/${src_name}.ll
        DEPENDS ${src_file}
        COMMENT "Compiling ${src_file} to LLVM IR"
    )
    list(APPEND LL_GENERATED ${CMAKE_BINARY_DIR}/${src_name}.ll)
endforeach()

# Link the .ll files into a final linked LLVM IR file
add_custom_command(
    OUTPUT ${LINKED_LL}
    COMMAND ${LLVM_LINK} ${LL_GENERATED} -S -o ${CMAKE_BINARY_DIR}/${LINKED_LL}
    DEPENDS ${LL_GENERATED}
    COMMENT "Linking LLVM IR files into ${LINKED_LL}"
)

# Create the executable from the linked LLVM IR
add_custom_command(
    OUTPUT ${EXECUTABLE}
    COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_BINARY_DIR}/${LINKED_LL} ${FUZZ_FLAGS} -o ${CMAKE_BINARY_DIR}/${EXECUTABLE}
    DEPENDS ${LINKED_LL}
    COMMENT "Compiling ${LINKED_LL} to executable ${EXECUTABLE}"
)

# Add a target for the executable
add_custom_target(${EXECUTABLE}_target ALL DEPENDS ${EXECUTABLE})

# Add a clean_generated target to clean up build files
add_custom_target(clean_generated
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${LL_GENERATED} ${CMAKE_BINARY_DIR}/${LINKED_LL} ${CMAKE_BINARY_DIR}/${EXECUTABLE}
    COMMENT "Cleaning up generated files"
)
