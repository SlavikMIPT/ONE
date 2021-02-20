/* mbed Microcontroller Library
 * Copyright (c) 2021 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 */
#include <mbed.h>
#include <iostream>
#include <luci_interpreter/Interpreter.h>
#include <luci/IR/Module.h>
#include <luci/Importer.h>
#include <iostream>
#include <loco/IR/DataTypeTraits.h>
#include "circlemodel.h"
#include "mio/circle/schema_generated.h"
// #include "stdex/Memory.h"

int main()
{
#ifdef TEST_LUCIINTERPRETER
  serial_logger.info("STM32F767 SystemCoreClock %d\n", SystemCoreClock);
  serial_logger.info("Model NET_0000.circle\n");

  std::vector<char> *buf = new std::vector<char>(
      circle_model_raw, circle_model_raw + sizeof(circle_model_raw) / sizeof(circle_model_raw[0]));
  std::vector<char> &model_data = *buf;
  // Verify flatbuffers
  flatbuffers::Verifier verifier{
      static_cast<const uint8_t *>(static_cast<void *>(model_data.data())), model_data.size()};
  serial_logger.info("circle::VerifyModelBuffer\n");
  if (!circle::VerifyModelBuffer(verifier))
  {
    serial_logger.info("ERROR: Failed to verify circle\n");
  }
  serial_logger.info("OK\n");
  // auto model = circle::GetModel(static_cast<const uint8_t *>(static_cast<void
  // *>(model_data.data()))); serial_logger.info("%s\n", model->description()->c_str());
  serial_logger.info("luci::Importer().importModule\n");

  auto module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (module == nullptr)
  {
    serial_logger.info("ERROR: Failed to load \n");
  }
  serial_logger.info("OK\n");

  serial_logger.info("luci_interpreter::Interpreter\n");

  auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module.get());
  auto nodes = module->graph()->nodes();
  auto nodes_count = nodes->size();
  serial_logger.info("nodes_count: %d\n", nodes_count);
  // Fill input tensors with some garbage
  while (true)
  {
    Timer t;
    for (int i = 0; i < nodes_count; ++i)
    {
      auto *node = dynamic_cast<luci::CircleNode *>(nodes->at(i));
      assert(node);
      if (node->opcode() == luci::CircleOpcode::CIRCLEINPUT)
      {
        auto *input_node = static_cast<luci::CircleInput *>(node);
        loco::GraphInput *g_input = module->graph()->inputs()->at(input_node->index());
        const loco::TensorShape *shape = g_input->shape();
        size_t data_size = 1;
        for (int d = 0; d < shape->rank(); ++d)
        {
          assert(shape->dim(d).known());
          data_size *= shape->dim(d).value();
        }
        data_size *= loco::size(g_input->dtype());
        std::vector<char> data(data_size);
        fill_in_tensor(data, g_input->dtype());

        interpreter->writeInputTensor(static_cast<luci::CircleInput *>(node), data.data(),
                                      data_size);
      }
    }
    t.start();
    interpreter->interpret();
    t.stop();
    serial_logger.info("\rFinished in %dus   ", t.read_us());
    ThisThread::sleep_for(10);
  }
#endif
  return 0;
}