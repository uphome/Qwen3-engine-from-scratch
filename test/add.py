import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector. 指向第一个输入向量的指针。
               y_ptr,  # *Pointer* to second input vector. 指向第二个输入向量的指针。
               output_ptr,  # *Pointer* to output vector. 指向输出向量的指针。
               n_elements,  # Size of the vector. 向量的大小。
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process. 每个程序应处理的元素数量。
               # NOTE: `constexpr` so it can be used as a shape value. 注意：`constexpr` 因此它可以用作形状值。
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # 有多个“程序”处理不同的数据。需要确定是哪一个程序：
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0. 使用 1D 启动网格，因此轴为 0。
    # This program will process inputs that are offset from the initial data.
    # 该程序将处理相对初始数据偏移的输入。
    # For instance, if you had a vector of length 256 and block_size of 64, the programs would each access the elements [0:64, 64:128, 128:192, 192:256].
    # 例如，如果有一个长度为 256, 块大小为 64 的向量，程序将各自访问 [0:64, 64:128, 128:192, 192:256] 的元素。
    # Note that offsets is a list of pointers:
    # 注意 offsets 是指针列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    # 创建掩码以防止内存操作超出边界访问。
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a multiple of the block size.
    # 从 DRAM 加载 x 和 y，如果输入不是块大小的整数倍，则屏蔽掉任何多余的元素。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    # 将 x + y 写回 DRAM。
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    # 需要预分配输出。
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # SPMD 启动网格表示并行运行的内核实例的数量。
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # 它类似于 CUDA 启动网格。它可以是 Tuple[int]，也可以是 Callable(metaparameters) -> Tuple[int]。
    # In this case, we use a 1D grid where the size is the number of blocks:
    # 在这种情况下，使用 1D 网格，其中大小是块的数量：
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    # 注意：
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - 每个 torch.tensor 对象都会隐式转换为其第一个元素的指针。
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - `triton.jit` 函数可以通过启动网格索引来获得可调用的 GPU 内核。
    #  - Don't forget to pass meta-parameters as keywords arguments.
    #  - 不要忘记以关键字参数传递元参数。
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still running asynchronously at this point.
    # 返回 z 的句柄，但由于 `torch.cuda.synchronize()` 尚未被调用，此时内核仍在异步运行。
    return output
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')