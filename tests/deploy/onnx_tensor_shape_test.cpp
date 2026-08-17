#include "isaaclab/algorithms/tensor_shape.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_throws(Callable&& callable, const char* message)
{
    try {
        callable();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_dynamic_batch_resolves_to_one()
{
    const auto input = isaaclab::onnx::resolve_batch_one_shape(
        {-1, 384}, "obs");
    require(input.dimensions == std::vector<std::int64_t>({1, 384}),
            "dynamic input batch must resolve to one");
    require(input.element_count == 384,
            "dynamic input element count must be 384");

    const auto output = isaaclab::onnx::resolve_batch_one_shape(
        {-1, 29}, "actions");
    require(output.dimensions == std::vector<std::int64_t>({1, 29}),
            "dynamic output batch must resolve to one");
    require(output.element_count == 29,
            "dynamic output element count must be 29");
}

void test_static_shape_remains_unchanged()
{
    const auto input = isaaclab::onnx::resolve_batch_one_shape(
        {1, 100}, "obs");
    require(input.dimensions == std::vector<std::int64_t>({1, 100}),
            "static Velocity shape must remain unchanged");
    require(input.element_count == 100,
            "static Velocity element count must remain 100");
}

void test_non_batch_dynamic_and_overflow_fail_closed()
{
    require_throws(
        [] {
            (void)isaaclab::onnx::resolve_batch_one_shape(
                {-1, -1}, "obs");
        },
        "non-batch dynamic dimensions must fail closed");
    require_throws(
        [] {
            (void)isaaclab::onnx::resolve_batch_one_shape({}, "obs");
        },
        "empty tensor shape must fail closed");
    require_throws(
        [] {
            (void)isaaclab::onnx::resolve_batch_one_shape(
                {1, std::numeric_limits<std::int64_t>::max(), 3}, "obs");
        },
        "tensor element-count overflow must fail closed");
}

void test_tensor_data_size_must_match_exactly()
{
    isaaclab::onnx::validate_tensor_data_size(384, 384, "obs");
    require_throws(
        [] {
            isaaclab::onnx::validate_tensor_data_size(383, 384, "obs");
        },
        "short observation must fail closed");
    require_throws(
        [] {
            isaaclab::onnx::validate_tensor_data_size(385, 384, "obs");
        },
        "oversized observation must fail closed");
}

void test_tensor_data_must_be_finite()
{
    isaaclab::onnx::validate_tensor_data_finite(
        std::vector<float>{0.0F, 1.0F, -2.0F}, "obs");
    require_throws(
        [] {
            isaaclab::onnx::validate_tensor_data_finite(
                std::vector<float>{
                    0.0F, std::numeric_limits<float>::quiet_NaN()},
                "obs");
        },
        "NaN observation must fail before ONNX inference");
    require_throws(
        [] {
            isaaclab::onnx::validate_tensor_data_finite(
                std::vector<float>{
                    0.0F, std::numeric_limits<float>::infinity()},
                "obs");
        },
        "infinite observation must fail before ONNX inference");
}

}  // namespace

int main()
{
    try {
        test_dynamic_batch_resolves_to_one();
        test_static_shape_remains_unchanged();
        test_non_batch_dynamic_and_overflow_fail_closed();
        test_tensor_data_size_must_match_exactly();
        test_tensor_data_must_be_finite();
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "all ONNX tensor-shape tests passed\n";
    return EXIT_SUCCESS;
}
