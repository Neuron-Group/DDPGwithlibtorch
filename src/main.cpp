#include "bits/stdc++.h"
#include "torch/torch.h"

using namespace torch::indexing;

int main(){
    torch::Tensor tensor = torch::tensor({1.0, 2.0, 3.0, 4.0});
    std::cout << tensor << std::endl;
    // 将 tensor 加载到 GPU 上
    tensor = tensor.to(torch::kCUDA);
    std::cout << tensor << std::endl;
    
    std::cout << "dim : " << tensor.dim() << std::endl;
    std::cout << "size_0 : " << tensor.size(0) << std::endl;
    // std::cout << "size_1 : " << tensor.size(1) << std::endl;
    std::cout << "sizes : " << tensor.sizes() << std::endl;
    std::cout << "numel : " << tensor.numel() << std::endl;
    std::cout << "dtype : " << tensor.dtype() << std::endl;
    std::cout << "scalar_type : " << tensor.scalar_type() << std::endl;
    std::cout << "device : " << tensor.device() << std::endl;

    // 利用下标访问元素
    auto tensor1 = torch::randn({1, 2, 3, 4});
    auto value = tensor1[0][1][2][2];
    std::cout << "value : " << value << std::endl;
    value = value.to(torch::kCUDA);
    value.print();
    std::cout << "value_float : " << value.item<float>() << std::endl;

    // value = value.item<float>();
    // std::cout << "value_float : " << value << std::endl;

    // delete value;

    value = tensor1.index({Slice(0, 1), Slice(), Slice(2, None), Slice(None, -1)});
    std::cout << "value : " << value << std::endl;
    value.print();

    // 修改 tensor1 中的值
    std::cout << tensor1 << std::endl;
    tensor1[0][1][1][1] = 2.5;
    std::cout << tensor1 << std::endl;

    return 0;
}