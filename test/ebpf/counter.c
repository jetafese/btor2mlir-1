#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
// #include <assert.h>

struct bpf_map_def SEC("maps") pkt_counter = {
    .type = BPF_MAP_TYPE_PERCPU_ARRAY,
    .key_size = sizeof(__u32),
    .value_size = sizeof(__u64),
    .max_entries = 1,
};

SEC("xdp")
int count_packets(struct __sk_buff *skb) {
    __u32 key = 0;
    __u64 *counter;

    counter = bpf_map_lookup_elem(&pkt_counter, &key);
    if (counter) {
        (*counter)++;
        if (*counter > 0) {
          bpf_get_prandom_u32();
        }
    }

    return 0;
}

char _license[] SEC("license") = "GPL";
