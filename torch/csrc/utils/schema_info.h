#pragma once

#include <ATen/core/alias_info.h>
#include <ATen/core/function_schema.h>
#include <iterator>

namespace torch {
namespace utils {

enum TORCH_API SchemaArgType { input, output };

/**
 * struct SchemaArgument
 *
 * Structure used to represent arguments or returns for a schema.
 */
struct TORCH_API SchemaArgument {
  SchemaArgType type;
  int index;
};

/**
 * class SchemaInfo
 *
 * Class that publicizes operator behavior (mutation, aliasing, special cases,
 * etc...)
 */

class TORCH_API SchemaInfo {
 public:
  SchemaInfo(c10::FunctionSchema schema) : schema_(schema) {}

  bool isMutating(int index) const;

  bool isMutating(c10::string_view name) const;

  bool areAliasing(const SchemaArgument& lhs, const SchemaArgument& rhs) const;

 private:
  c10::FunctionSchema schema_;
};
} // namespace utils
} // namespace torch
