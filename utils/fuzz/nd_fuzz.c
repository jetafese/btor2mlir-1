/** Definitions of nd() functions for fuzzing */

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <setjmp.h>

/** Global access to fuzz data */
uint8_t *g_fuzz_data;
size_t g_fuzz_data_size;
uint8_t *g_fuzz_data_iterator;

/** jmp environment */
jmp_buf g_jmp_buf;

#define UPDATE_FUZZ_ITERATOR(TYPE)                                             \
  if (g_fuzz_data_iterator + sizeof(TYPE) - g_fuzz_data >= g_fuzz_data_size) { \
    longjmp(g_jmp_buf, 1);                                             \
  }

// bool nd_bool(void) {
//   int tmp;
//   UPDATE_FUZZ_ITERATOR(bool)
//   memcpy(&tmp, g_fuzz_data_iterator, sizeof(bool));
//   g_fuzz_data_iterator += sizeof(bool);
//   return (tmp > 0);
// }

// int nd_int(void) {
//   int res;

//   UPDATE_FUZZ_ITERATOR(int);
//   memcpy(&res, g_fuzz_data_iterator, sizeof(int));
//   g_fuzz_data_iterator += sizeof(int);

//   return res;
// }

// uint8_t nd_uint8_t(void) {
//   uint8_t res;

//   UPDATE_FUZZ_ITERATOR(uint8_t);
//   memcpy(&res, g_fuzz_data_iterator, sizeof(uint8_t));
//   g_fuzz_data_iterator += sizeof(uint8_t);

//   return res;
// }

// uint16_t nd_uint16_t(void) {
//   uint16_t res;

//   UPDATE_FUZZ_ITERATOR(uint16_t);
//   memcpy(&res, g_fuzz_data_iterator, sizeof(uint16_t));
//   g_fuzz_data_iterator += sizeof(uint16_t);

//   return res;
// }

uint32_t nd_bv32(void) {
  uint32_t res;

  UPDATE_FUZZ_ITERATOR(uint32_t);
  memcpy(&res, g_fuzz_data_iterator, sizeof(uint32_t));
  g_fuzz_data_iterator += sizeof(uint32_t);
  fprintf(stdout, "[sea] nd_bv32: %d\n", res);
  fflush(stdout);

  return 1;
}

// uint64_t nd_uint64_t(void) {
//   uint64_t res;

//   UPDATE_FUZZ_ITERATOR(uint64_t);
//   memcpy(&res, g_fuzz_data_iterator, sizeof(uint64_t));
//   g_fuzz_data_iterator += sizeof(uint64_t);

//   return res;
// }

// size_t nd_size_t(void) {
//   size_t res;

//   UPDATE_FUZZ_ITERATOR(size_t);
//   memcpy(&res, g_fuzz_data_iterator, sizeof(size_t));
//   g_fuzz_data_iterator += sizeof(size_t);

//   return res;
// }

// void *nd_voidp(void) {
//   void *res;

//   UPDATE_FUZZ_ITERATOR(res);
//   memcpy(&res, g_fuzz_data_iterator, sizeof(res));
//   g_fuzz_data_iterator += sizeof(res);

//   return res;
// }

// void memhavoc(void *ptr, size_t size) {
//   if (!ptr) {
//     return;
//   }
//   if (g_fuzz_data_iterator + size - g_fuzz_data >= g_fuzz_data_size) {
//     longjmp(g_jmp_buf, 1);
//   }

//   memcpy(ptr, g_fuzz_data_iterator, size);
//   g_fuzz_data_iterator += size;
// }

void __VERIFIER_assert(bool v, int property) {
  fprintf(stdout, "CHECK ASSERT\n");
  if (v) {
    fprintf(stdout, "[sea] __VERIFIER_assert was called for property: %d\n", property);
  }
  fflush(stdout);

  return;
}

void __SEA_assume(bool v) {
  // -- if assumption fails, jump back to LLVMFuzzerTestOneInput
  if (!v) {
    longjmp(g_jmp_buf, 1);
  }
}

void __VERIFIER_assume(int v) {
  __SEA_assume(v);
}

void __VERIFIER_error() {
  fprintf(stdout, "[sea] __VERIFIER_error was executed\n");
  fflush(stdout);
  assert(0);
}

void __TRACKER() {
  fprintf(stdout, "finished another loop\n");
  fflush(stdout);
}

void btor2mlir_print_input_num(unsigned num, unsigned value, unsigned width) { 
  fprintf(stdout, "input, %d, %d, %d\n", num, value, width);
  fflush(stdout);
}

void btor2mlir_print_state_num(unsigned num, unsigned value, unsigned width) { 
  fprintf(stdout, "state, %d, %d, %d\n", num, value, width);
  fflush(stdout);
}

void btor2mlir_print_array_state_num(unsigned num, unsigned index, unsigned value, unsigned width) { 
  fprintf(stdout, "state, %d, %d, %d, %d\n", num, index, value, width);
  fflush(stdout);
}

/** expected entry of verification harness */
extern int _main(void);

int LLVMFuzzerTestOneInput(uint8_t *Data, size_t Size) {
  g_fuzz_data = Data;
  g_fuzz_data_size = Size;
  g_fuzz_data_iterator = g_fuzz_data;

  if (setjmp(g_jmp_buf)) {
    // assumption failed, abort
    return 0;
  }
  _main();

  return 0;
}

// bool nd_malloc_is_fail(void) { return nd_bool(); }

// int nondet_compare(const void* a, const void *b) {
//   return nd_int();
// }

// /* nd_linked_list_node - special function for returning a symbolic value of
//  * the linked list node pointer.
// */
// struct aws_linked_list_node *nd_linked_list_node(void) {
//   return (struct aws_linked_list_node *)nd_voidp();
// }
