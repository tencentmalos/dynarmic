/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#include <random>

#include <catch2/catch_test_macros.hpp>
#include <oaknut/oaknut.hpp>

#include "testenv.h"

using namespace Dynarmic;

TEST_CASE("A64: bit select preserves all overlap patterns and upper lanes", "[a64][bit-select]") {
    // All five equality partitions of the three guest registers. Host liveness
    // is also exercised by retaining every other register and a copy of Vd.
    constexpr int overlaps[][3]{{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
    std::mt19937_64 random{0xB17B1FB5ULL};
    for (const bool wide : {false, true}) {
        for (int op = 0; op < 3; ++op) {
            for (const auto& regs : overlaps) {
                for (const bool live_alias : {false, true}) {
                    CAPTURE(wide, op, regs[0], regs[1], regs[2], live_alias);
                    const int d = regs[0], n = regs[1], m = regs[2];
                    A64TestEnv env;
                    oaknut::VectorCodeGenerator code{env.code_mem, nullptr};
                    if (live_alias)
                        code.MOV(oaknut::QReg{31}.B16(), oaknut::QReg{d}.B16());
                    const auto emit = [&](auto dst, auto src, auto mask) {
                        switch (op) {
                        case 0:
                            code.BSL(dst, src, mask);
                            break;
                        case 1:
                            code.BIT(dst, src, mask);
                            break;
                        case 2:
                            code.BIF(dst, src, mask);
                            break;
                        }
                    };
                    if (wide)
                        emit(oaknut::QReg{d}.B16(), oaknut::QReg{n}.B16(), oaknut::QReg{m}.B16());
                    else
                        emit(oaknut::DReg{d}.B8(), oaknut::DReg{n}.B8(), oaknut::DReg{m}.B8());
                    // Consume the retained identity after the destructive operation.
                    if (live_alias)
                        code.EOR(oaknut::QReg{30}.B16(), oaknut::QReg{31}.B16(), oaknut::QReg{d}.B16());
                    code.B(std::ptrdiff_t{0});
                    A64::Jit jit{A64::UserConfig{&env}};
                    for (int sample = 0; sample < 66; ++sample) {
                        std::array<Vector, 32> initial;
                        for (auto& v : initial)
                            v = {random(), random()};
                        // Force all-zero and all-one masks, including aliases.
                        if (sample < 2)
                            initial[op == 0 ? d : m] = sample ? Vector{~u64{0}, ~u64{0}} : Vector{};
                        auto expected = initial;
                        if (live_alias)
                            expected[31] = initial[d];
                        // Independent bitwise scalar oracle, including the architectural
                        // zeroing of the upper half of a 64-bit SIMD destination.
                        expected[d] = {};
                        for (int bit = 0; bit < (wide ? 128 : 64); ++bit) {
                            const int lane = bit / 64, shift = bit % 64;
                            const bool select = ((initial[op == 0 ? d : m][lane] >> shift) & 1) != 0;
                            const int source = op == 0 ? (select ? n : m) : op == 1 ? (select ? n : d)
                                                                                    : (select ? d : n);
                            expected[d][lane] |= ((initial[source][lane] >> shift) & 1) << shift;
                        }
                        if (live_alias)
                            for (int lane = 0; lane < 2; ++lane)
                                expected[30][lane] = expected[31][lane] ^ expected[d][lane];
                        jit.SetVectors(initial);
                        jit.SetPC(0);
                        env.ticks_left = env.code_mem.size() - 1;
                        jit.Run();
                        REQUIRE(jit.GetVectors() == expected);
                        REQUIRE(env.interrupts.empty());
                    }
                }
            }
        }
    }
}
