#pragma once
#include <torch/torch.h>

class PerceptronTorch : public torch::nn::Module {
private:
    torch::nn::Linear linear{nullptr};

public:
    PerceptronTorch(int input_size, int output_size = 1) {
        linear = register_module("linear", torch::nn::Linear(input_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = linear->forward(x);
        return out >= 0;
    }
};
