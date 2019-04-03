#include <libgpu/cuda/cu/opencl_translator.cu>

#include "../cl/aplusb.cl"

void cuda_aplusb(const gpu::WorkSize &workSize, cudaStream_t stream,
                 const float* a, const float* b, float* c, unsigned int n) {
    aplusb<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a, b, c, n);
    CUDA_CHECK_KERNEL(stream);
}

