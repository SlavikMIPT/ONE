/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "Builders.h"
#include "kernels/Utils.h"

#include <cstring>

namespace luci_interpreter
{

void configure_kernel_CircleIf(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  auto *runtime_module = runtime_graph->getRuntimeModule();

  const auto *options = cur_op->builtin_options_as_IfOptions();

  const auto cond_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  const auto then_subgraph_index = options->then_subgraph_index();
  const auto else_subgraph_index = options->else_subgraph_index();

  assert(cond_index != -1);
  assert(output_index != -1);

  assert(then_subgraph_index != -1);
  assert(else_subgraph_index != -1);

  const auto cond = runtime_graph->getCircleTensorByIndex(cond_index);
  LUCI_INTERPRETER_CHECK(Tensor::element_type(cond) == DataType::BOOL);
  LUCI_INTERPRETER_CHECK(Tensor::num_elements(cond) == 1);

  const auto output = runtime_graph->getCircleTensorByIndex(output_index);
  auto *then_subgraph = runtime_module->getRuntimeGraphAt(then_subgraph_index);
  auto *else_subgraph = runtime_module->getRuntimeGraphAt(else_subgraph_index);
  for (RuntimeGraph *graph : {then_subgraph, else_subgraph})
  {
    (void)graph;
    LUCI_INTERPRETER_CHECK(graph->getNumOfInputTensors() ==
                           runtime_graph->getNumOfInputTensors() - 1);
    LUCI_INTERPRETER_CHECK(graph->getNumOfOutputTensors() ==
                           runtime_graph->getNumOfOutputTensors());
  }
}

void execute_kernel_CircleIf(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  auto *runtime_module = runtime_graph->getRuntimeModule();

  const auto *options = cur_op->builtin_options_as_IfOptions();
  const auto cond_index = cur_op->inputs()->operator[](0);
  const auto output_index = cur_op->outputs()->operator[](0);

  const auto then_subgraph_index = options->then_subgraph_index();
  const auto else_subgraph_index = options->else_subgraph_index();

  auto *then_subgraph = runtime_module->getRuntimeGraphAt(then_subgraph_index);
  auto *else_subgraph = runtime_module->getRuntimeGraphAt(else_subgraph_index);

  const auto cond = runtime_graph->getCircleTensorByIndex(cond_index);
  const auto output = runtime_graph->getCircleTensorByIndex(output_index);

  const uint8_t *cond_data = runtime_graph->getDataByTensor(cond);
  const bool cond_value = kernels::getTensorData<bool>(cond_data)[0];

  RuntimeGraph *active_graph = cond_value ? then_subgraph : else_subgraph;

  // Copy kernel inputs to active graph inputs.
  for (size_t i = 0; i < runtime_graph->getNumOfInputTensors() - 1; ++i)
  {
    auto active_input = active_graph->getInputTensorByIndex(i);
    auto runtime_input = runtime_graph->getInputTensorByIndex(i);

    LUCI_INTERPRETER_CHECK(Tensor::element_type(active_input) ==
                           (Tensor::element_type(runtime_input)));

    //    active_graph->getInputTensorByIndex(i)->resize(
    //      runtime_graph->getInputTensorByIndex(i)->shape());

    const int32_t num_elements = Tensor::num_elements(runtime_input);
    const std::size_t element_size = size(Tensor::element_type(runtime_input));
    //    // TODO: Think about how allocate memory for output in main graph
    //    active_graph->configureAllocations(active_input);

    std::memcpy(runtime_graph->getDataByTensor(active_input),
                runtime_graph->getDataByTensor(runtime_input), num_elements * element_size);
  }
  active_graph->execute();

  // Copy graph outputs to kernel outputs.
  for (size_t i = 0; i < runtime_graph->getNumOfOutputTensors(); ++i)
  {
    auto active_output = active_graph->getOutputTensorByIndex(i);
    auto runtime_output = runtime_graph->getOutputTensorByIndex(i);

    LUCI_INTERPRETER_CHECK(Tensor::element_type(active_output) ==
                           (Tensor::element_type(runtime_output)));

    //    active_graph->getInputTensorByIndex(i)->resize(
    //      runtime_graph->getInputTensorByIndex(i)->shape());

    const int32_t num_elements = Tensor::num_elements(runtime_output);
    const std::size_t element_size = size(Tensor::element_type(runtime_output));
    //    // TODO: Think about how allocate memory for output in main graph
    //    active_graph->configureAllocations(active_input);

    std::memcpy(runtime_graph->getDataByTensor(runtime_output),
                runtime_graph->getDataByTensor(active_output), num_elements * element_size);
  }
}

} // namespace luci_interpreter
