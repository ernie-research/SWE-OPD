#!/bin/bash
# Stage 10: synthesize a self-contained CUDA 12.9 CUDA_HOME at $CUDA_SHIM.
#
# WHY a shim: the container ships CUDA 13.0 ONLY, but torch is 2.11.0+cu129, so
# every CUDA extension (flash-attn / apex / TE) must be compiled by a 12.9 nvcc.
#
# WHY not pure pip wheels: `nvidia-cuda-nvcc-cu12` on pypi ships ONLY ptxas --
# verified against 12.6.85 / 12.8.93 / 12.9.41 / 12.9.86, every one of them has
# exactly `nvidia/cuda_nvcc/bin/ptxas` and nothing else. There is no nvcc, cicc,
# cudafe++, nvlink or fatbinary in any pip package. So the nvcc FRONTEND comes
# from NVIDIA's redistributable archive (same bits, same 12.9.86 build, just a
# tarball instead of a wheel) and everything else (headers + .so) comes from the
# wheels already installed in the venv. Nothing is installed system-wide;
# /usr/local/cuda-13.0 is left untouched.
set -ex
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SP="$VENV_DIR/lib/python3.12/site-packages/nvidia"
REDIST_DIR="$BUILD_DIR/redist"
NVCC_VER=12.9.86
NVCC_TARBALL="cuda_nvcc-linux-x86_64-${NVCC_VER}-archive.tar.xz"
NVCC_URL="https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/${NVCC_TARBALL}"

mkdir -p "$REDIST_DIR"
if [[ ! -d "$REDIST_DIR/cuda_nvcc-linux-x86_64-${NVCC_VER}-archive" ]]; then
   [[ -f "$REDIST_DIR/$NVCC_TARBALL" ]] || curl -fL --retry 3 -o "$REDIST_DIR/$NVCC_TARBALL" "$NVCC_URL"
   tar -xf "$REDIST_DIR/$NVCC_TARBALL" -C "$REDIST_DIR"
fi
NVCC_SRC="$REDIST_DIR/cuda_nvcc-linux-x86_64-${NVCC_VER}-archive"
test -x "$NVCC_SRC/bin/nvcc"

# --- build the shim tree -----------------------------------------------------
rm -rf "$CUDA_SHIM"
mkdir -p "$CUDA_SHIM"/{bin,include,lib64,lib64/stubs}

# nvcc frontend: bin/ (nvcc, cicc, cudafe++, nvlink, fatbinary, ptxas,
# nvcc.profile) + nvvm/ + its own crt headers. Real copies, not links, so
# nvcc's _HERE_/TOP resolution lands inside the shim.
cp -a "$NVCC_SRC/bin/." "$CUDA_SHIM/bin/"
cp -a "$NVCC_SRC/nvvm" "$CUDA_SHIM/nvvm"
cp -a "$NVCC_SRC/include/." "$CUDA_SHIM/include/"

# Headers from the venv wheels. cuda_runtime carries cuda.h/cuda_runtime.h,
# cuda_cccl carries cub/thrust/libcudacxx (apex + TE need these).
for d in cuda_runtime cuda_cccl cuda_profiler_api cuda_cuxxfilt cublas cuda_nvrtc \
         nvjitlink nvfatbin nvtx cufft curand cusolver cusparse cuda_cupti nvml_dev cudnn; do
   [[ -d "$SP/$d/include" ]] && cp -a "$SP/$d/include/." "$CUDA_SHIM/include/"
done

# Libraries: symlink (cublas alone is ~700MB, no point duplicating).
for d in cuda_runtime cuda_cccl cublas cuda_nvrtc nvjitlink nvfatbin nvtx \
         cufft curand cusolver cusparse cuda_cupti cudnn; do
   for sub in lib lib64; do
      [[ -d "$SP/$d/$sub" ]] || continue
      find "$SP/$d/$sub" -maxdepth 1 -type f -exec ln -sf {} "$CUDA_SHIM/lib64/" \;
      # cupti/nvperf live in extras in a real toolkit; flat lib64 is fine here.
   done
done
rm -f "$CUDA_SHIM/lib64/__init__.py"

# The wheels ship only SONAME-versioned libs (libcudart.so.12, ...). A real
# toolkit also has the unversioned dev symlink that `-lcudart` / `-lcublas`
# resolve against, so synthesize them.
for f in "$CUDA_SHIM"/lib64/*.so.*; do
   [[ -e "$f" ]] || continue
   base=$(basename "$f")
   dev="${base%%.so.*}.so"
   [[ -e "$CUDA_SHIM/lib64/$dev" ]] || ln -sf "$base" "$CUDA_SHIM/lib64/$dev"
done

# libcuda.so stub for link-time -lcuda (driver API). The 13.0 stub is ABI-forward
# compatible for linking; at runtime the real driver libcuda.so.1 is used.
ln -sf /usr/local/cuda-13.0/lib64/stubs/libcuda.so "$CUDA_SHIM/lib64/stubs/libcuda.so"

# targets/x86_64-linux/{include,lib}: some builds (apex, cmake's FindCUDAToolkit)
# probe this layout instead of $CUDA_HOME/{include,lib64}.
mkdir -p "$CUDA_SHIM/targets/x86_64-linux"
ln -sfn "$CUDA_SHIM/include" "$CUDA_SHIM/targets/x86_64-linux/include"
ln -sfn "$CUDA_SHIM/lib64" "$CUDA_SHIM/targets/x86_64-linux/lib"
ln -sfn "$CUDA_SHIM/lib64" "$CUDA_SHIM/lib"

# --- validate ----------------------------------------------------------------
# shellcheck disable=SC1091
activate_venv
command -v nvcc
nvcc --version
test "$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')" = "12.9"

# Real compile+link+run for sm90 (this is what stage 20+ depends on).
TMPD=$(mktemp -d)
cat > "$TMPD/probe.cu" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(float* o){ o[threadIdx.x] = threadIdx.x * 2.0f; }
int main(){
  float* d; cudaMalloc(&d, 4*sizeof(float));
  k<<<1,4>>>(d);
  float h[4]; cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
  printf("shim_ok %.1f %.1f %.1f %.1f\n", h[0],h[1],h[2],h[3]);
  return 0;
}
CU
nvcc -arch=sm_90 -I"$CUDA_SHIM/include" -L"$CUDA_SHIM/lib64" \
     "$TMPD/probe.cu" -o "$TMPD/probe" -lcudart
"$TMPD/probe"

# thrust/cub header path (apex + TE compile these) and the torch extension path.
cat > "$TMPD/probe_cccl.cu" <<'CU'
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <cstdio>
int main(){ thrust::device_vector<int> v(4, 7); printf("cccl_ok %d\n", (int)v[3]); return 0; }
CU
nvcc -arch=sm_90 -std=c++17 -I"$CUDA_SHIM/include" -L"$CUDA_SHIM/lib64" \
     "$TMPD/probe_cccl.cu" -o "$TMPD/probe_cccl" -lcudart
"$TMPD/probe_cccl"
rm -rf "$TMPD"

python -c "
import torch.utils.cpp_extension as e
print('torch CUDA_HOME =', e.CUDA_HOME)
print('nvcc ver        =', e._get_cuda_arch_flags.__module__ and 'ok')
"
echo "CUDA_SHIM READY: $CUDA_SHIM"
