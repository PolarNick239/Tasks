#include <libgpu/cuda/cu/opencl_translator.cu>

#include "../cl/aplusb.cl"

void cuda_aplusb(const gpu::WorkSize &workSize,
                 const float* a, const float* b, float* c, unsigned int n) {
    aplusb<<<workSize.cuGridSize(), workSize.cuBlockSize()>>>(a, b, c, n);
}

