#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <random> 

// initialize random number generator
std::random_device rd;
std::mt19937 gen(rd());

extern "C" {

    long bpf_get_current_comm(void *buf, uint32_t size_of_buf);

    int64_t nd_int() {
        std::uniform_int_distribution<int64_t> dis;
        return dis(gen);
    }

    long nd_long() {
        std::uniform_int_distribution<long> dis;
        return dis(gen);
    }

    uint64_t nd_u64() {
        std::uniform_int_distribution<uint64_t> dis;
        return dis(gen);
    }

    uint32_t nd_u32() {
        std::uniform_int_distribution<uint32_t> dis;
        return dis(gen);
    }

    int64_t memhavoc(void *ptr, int size) {
        if (ptr == nullptr){
            return -1; 
        }
        memset(ptr, nd_int(), size); 
        return 0; 
    }

}

// Passing test case
void test_bpf_get_current_comm_pass() {
    const uint32_t buffer_size = 64; 
    void *buf = malloc(buffer_size);
    assert(buf != nullptr); 

    memset(buf, 0, buffer_size); 

    long result = bpf_get_current_comm(buf, buffer_size);

    std::cout << "bpf_get_current_comm (passing) result: " << result << std::endl;

    free(buf);
}

// Failing test case
void test_bpf_get_current_comm_fail() {
    const uint32_t buffer_size = 64; 
    void *buf = malloc(buffer_size);
    assert(buf != nullptr); 
    memhavoc(buf, buffer_size); // fill buffer with random data (to make the test fail
    long result = bpf_get_current_comm(buf, buffer_size + 1); // should fail 
    std::cout << "bpf_get_current_comm (failing) result: " << result << std::endl;
    free(buf);
}

uint32_t min(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

// Fuzzer entry point
extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
    if (size < sizeof(uint32_t)) {
        return 0;
    }

    uint32_t buffer_size = *(reinterpret_cast<const uint32_t*>(data));
    buffer_size = buffer_size % 1024;
    void *buf = malloc(buffer_size);
    if (buf == nullptr) {
        return 0;
    }

    memset(buf, 0, buffer_size);
    memcpy(buf, data + sizeof(uint32_t), min(buffer_size, size - sizeof(uint32_t)));

    bpf_get_current_comm(buf, buffer_size);

    free(buf);
    return 0;
}

int main(int argc, char** argv) {
    std::cout << "Running passing test for bpf_get_current_comm..." << std::endl;
    test_bpf_get_current_comm_pass();
    std::cout << "Passing test completed." << std::endl;

    std::cout << "Running failing test for bpf_get_current_comm..." << std::endl;
    test_bpf_get_current_comm_fail();
    std::cout << "Failing test completed." << std::endl;
    return 0;
}
