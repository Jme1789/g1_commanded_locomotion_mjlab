#pragma once

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace isaaclab::onnx {

struct ResolvedTensorShape
{
    std::vector<std::int64_t> dimensions;
    std::size_t element_count{0};
};

inline ResolvedTensorShape resolve_batch_one_shape(
    const std::vector<std::int64_t>& model_shape,
    const std::string& tensor_name)
{
    if (model_shape.empty()) {
        throw std::runtime_error(
            "ONNX tensor '" + tensor_name + "' has an empty shape");
    }

    ResolvedTensorShape resolved{model_shape, 1};
    if (resolved.dimensions.front() <= 0) {
        resolved.dimensions.front() = 1;
    }

    for (std::size_t index = 0; index < resolved.dimensions.size(); ++index) {
        const auto dimension = resolved.dimensions[index];
        if (dimension <= 0) {
            throw std::runtime_error(
                "ONNX tensor '" + tensor_name + "' dimension " +
                std::to_string(index) + " must be positive");
        }
        const auto unsigned_dimension = static_cast<std::uint64_t>(dimension);
        if (unsigned_dimension >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()) ||
            resolved.element_count >
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(unsigned_dimension)) {
            throw std::runtime_error(
                "ONNX tensor '" + tensor_name + "' element count overflows");
        }
        resolved.element_count *= static_cast<std::size_t>(unsigned_dimension);
    }
    return resolved;
}

inline void validate_tensor_data_size(
    std::size_t actual,
    std::size_t expected,
    const std::string& tensor_name)
{
    if (actual != expected) {
        throw std::runtime_error(
            "ONNX tensor '" + tensor_name + "' expected " +
            std::to_string(expected) + " values, got " +
            std::to_string(actual));
    }
}

inline void validate_tensor_data_finite(
    const std::vector<float>& values,
    const std::string& tensor_name)
{
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error(
                "ONNX tensor '" + tensor_name +
                "' contains a non-finite value at index " +
                std::to_string(index));
        }
    }
}

}  // namespace isaaclab::onnx
