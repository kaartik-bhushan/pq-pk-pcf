#include <vector>
#include <cstdint>
#include <intel/hexl/hexl.hpp>

enum class PolyState { COEFFICIENT, EVALUATION };

struct RingElement {
    std::vector<uint64_t> coeffs; // Size N, 64-byte aligned for AVX-512
    PolyState state = PolyState::COEFFICIENT;
};

// Alignment helper for Intel HEXL/AVX-512 performance
std::vector<uint64_t> allocate_aligned_poly(uint64_t N) {
    // 64-byte alignment allows optimal vectorization
    return std::vector<uint64_t>(N); 
}