import pytest

from parser import parse_mlir

SAMPLE_IR = """// -----// IR Dump Before TritonGPUPlanCTAPass (triton-nvidia-gpu-plan-cta) ('builtin.module' operation) //----- //
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#loc = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0)
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:89", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0)) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c1024_i32 : i32 loc(#loc3)
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked> loc(#loc4)
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked> loc(#loc5)
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked> loc(#loc5)
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked> loc(#loc6)
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked> loc(#loc6)
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc7)
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc7)
    %9 = triton_gpu.convert_layout %8 : tensor<1024x!tt.ptr<f32>, #blocked> -> tensor<1024x!tt.ptr<f32>, #blocked1> loc(#loc8)
    %10 = triton_gpu.convert_layout %6 : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1> loc(#loc8)
    %11 = tt.load %9, %10 : tensor<1024x!tt.ptr<f32>, #blocked1> loc(#loc8)
    %12 = triton_gpu.convert_layout %11 : tensor<1024xf32, #blocked1> -> tensor<1024xf32, #blocked> loc(#loc8)
    %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc9)
    %14 = tt.addptr %13, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc9)
    %15 = triton_gpu.convert_layout %14 : tensor<1024x!tt.ptr<f32>, #blocked> -> tensor<1024x!tt.ptr<f32>, #blocked1> loc(#loc10)
    %16 = triton_gpu.convert_layout %6 : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1> loc(#loc10)
    %17 = tt.load %15, %16 : tensor<1024x!tt.ptr<f32>, #blocked1> loc(#loc10)
    %18 = triton_gpu.convert_layout %17 : tensor<1024xf32, #blocked1> -> tensor<1024xf32, #blocked> loc(#loc10)
    %19 = arith.addf %12, %18 : tensor<1024xf32, #blocked> loc(#loc11)
    %20 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc12)
    %21 = tt.addptr %20, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc12)
    %22 = triton_gpu.convert_layout %21 : tensor<1024x!tt.ptr<f32>, #blocked> -> tensor<1024x!tt.ptr<f32>, #blocked1> loc(#loc13)
    %23 = triton_gpu.convert_layout %19 : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1> loc(#loc13)
    %24 = triton_gpu.convert_layout %6 : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1> loc(#loc13)
    tt.store %22, %23, %24 : tensor<1024x!tt.ptr<f32>, #blocked1> loc(#loc13)
    tt.return loc(#loc14)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":37:24)
#loc3 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":42:24)
#loc4 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":43:41)
#loc5 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":43:28)
#loc6 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":45:21)
#loc7 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":48:24)
#loc8 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":48:16)
#loc9 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":49:24)
#loc10 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":49:16)
#loc11 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":50:17)
#loc12 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":52:26)
#loc13 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":52:35)
#loc14 = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":52:4)
"""

def test_parser():
    sections = parse_mlir(SAMPLE_IR)
    assert len(sections) == 14
    for loc_num, section in sections.items():
        assert "file_info" in section
        assert "code" in section
        assert len(section["code"]) > 0
        assert "/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py" in section["file_info"] or section["file_info"] == "unknown"

    assert sections["1"]["file_info"] == "unknown"
    assert sections["7"]["code"][0] == "%7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc7)"
    assert sections["8"]["code"][0] == "%9 = triton_gpu.convert_layout %8 : tensor<1024x!tt.ptr<f32>, #blocked> -> tensor<1024x!tt.ptr<f32>, #blocked1> loc(#loc8)"


if __name__ == "__main__":
    pytest.main()