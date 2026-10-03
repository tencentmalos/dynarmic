/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#include <array>
#include <random>
#include <unordered_map>

#include <catch2/catch_test_macros.hpp>

#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt/passes.h"

using namespace Dynarmic;

TEST_CASE("IR: bit select fallback preserves per-bit semantics", "[ir][bit-select-polyfill]") {
    using Vector = std::array<u64, 2>;
    IR::Inst mask{IR::Opcode::VectorNot}, yes{IR::Opcode::VectorNot}, no{IR::Opcode::VectorNot};
    IR::Block block{IR::LocationDescriptor{0}};
    block.AppendNewInst(IR::Opcode::VectorBitSelect, {IR::Value{&mask}, IR::Value{&yes}, IR::Value{&no}});
    auto* result = &block.back();
    Optimization::PolyfillPass(block, {.vector_bit_select = true});
    std::mt19937_64 random{0x51EC7};
    for (int sample = 0; sample < 258; ++sample) {
        std::unordered_map<const IR::Inst*, Vector> values{{&mask, {random(), random()}}, {&yes, {random(), random()}}, {&no, {random(), random()}}};
        if (sample < 2)
            values[&mask] = sample ? Vector{~u64{0}, ~u64{0}} : Vector{};
        Vector expected{};
        for (int bit = 0; bit < 128; ++bit) {
            const int lane = bit / 64, shift = bit % 64;
            const auto* source = (values[&mask][lane] >> shift) & 1 ? &yes : &no;
            expected[lane] |= ((values[source][lane] >> shift) & 1) << shift;
        }
        for (const auto& inst : block) {
            REQUIRE(inst.GetOpcode() != IR::Opcode::VectorBitSelect);
            const auto a = values.at(inst.GetArg(0).GetInst());
            if (inst.GetOpcode() == IR::Opcode::Identity) {
                values[&inst] = a;
                continue;
            }
            const auto b = values.at(inst.GetArg(1).GetInst());
            auto& output = values[&inst];
            for (int lane = 0; lane < 2; ++lane) {
                switch (inst.GetOpcode()) {
                case IR::Opcode::VectorEor:
                    output[lane] = a[lane] ^ b[lane];
                    break;
                case IR::Opcode::VectorAnd:
                    output[lane] = a[lane] & b[lane];
                    break;
                default:
                    FAIL("Unexpected opcode in logical fallback");
                }
            }
        }
        REQUIRE(values.at(result) == expected);
    }
}
