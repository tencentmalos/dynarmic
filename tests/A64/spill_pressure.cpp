/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#include <array>
#include <chrono>
#include <cstdio>

#include <catch2/catch_test_macros.hpp>
#include <mcl/macro/architecture.hpp>
#include <oaknut/oaknut.hpp>

#include "testenv.h"

#ifdef MCL_ARCHITECTURE_ARM64
#    include "dynarmic/backend/arm64/a64_address_space.h"
#    include "dynarmic/backend/arm64/a64_jitstate.h"
#    include "dynarmic/backend/arm64/stack_layout.h"
#endif

using namespace Dynarmic;
using namespace oaknut::util;

namespace {
constexpr struct Workload {
    const char* name;
    int registers;
    int rounds;
} workloads[]{{"small", 3, 1}, {"medium", 22, 1}, {"pressure", 30, 4}};

struct PressureProgram {
    std::array<u64, 31> initial_gpr{}, expected_gpr{};
    std::array<Vector, 32> initial_vec{}, expected_vec{};
    explicit PressureProgram(A64TestEnv& env, const Workload& workload) {
        for (size_t i = 0; i < initial_gpr.size(); ++i)
            initial_gpr[i] = 0x123456789ABCDEF0ULL * (i + 1);
        for (size_t i = 0; i < initial_vec.size(); ++i)
            initial_vec[i] = {0xF0EDCB123456789AULL * (i + 1), 0xA123B456C789D0EFULL * (i + 1)};
        expected_gpr = initial_gpr;
        expected_vec = initial_vec;
        oaknut::VectorCodeGenerator code{env.code_mem, nullptr};
        const int count = workload.registers;
        for (int r = 0; r < count; ++r) {
            code.ADD(oaknut::XReg{r}, oaknut::XReg{r}, r + 1);
            expected_gpr[r] += r + 1;
        }
        for (int round = 0; round < workload.rounds; ++round) {
            for (int r = 0; r < count; ++r) {
                const int next = (r + 1) % count;
                code.EOR(oaknut::XReg{r}, oaknut::XReg{r}, oaknut::XReg{next});
                expected_gpr[r] ^= expected_gpr[next];
                code.EOR(oaknut::QReg{r}.B16(), oaknut::QReg{r}.B16(), oaknut::QReg{next}.B16());
                expected_vec[r][0] ^= expected_vec[next][0];
                expected_vec[r][1] ^= expected_vec[next][1];
            }
        }
        code.B(std::ptrdiff_t{0});  // A separate terminal self-loop stops at the tick budget.
    }
    void Reset(A64::Jit& jit, A64TestEnv& env) const {
        jit.SetRegisters(initial_gpr);
        jit.SetVectors(initial_vec);
        jit.SetPC(0);
        env.ticks_left = env.code_mem.size() - 1;
    }
    void Check(const A64::Jit& jit) const {
        REQUIRE(jit.GetRegisters() == expected_gpr);
        REQUIRE(jit.GetVectors() == expected_vec);
    }
};

#ifdef MCL_ARCHITECTURE_ARM64
void ReportCodeSize(A64TestEnv& env, const char* workload) {
    using namespace Backend::Arm64;
    const A64::UserConfig user_config{&env};
    A64AddressSpace address_space{user_config};
    Backend::ExceptionHandler exception_handler;
    FastmemManager fastmem{exception_handler};
    EmitConfig config{};
    config.optimizations = user_config.optimizations;
    config.state_nzcv_offset = offsetof(A64JitState, cpsr_nzcv);
    config.state_fpsr_offset = offsetof(A64JitState, fpsr);
    // This probe counts the integer/SIMD block body, excluding the common
    // dispatcher/prelude and timing/terminal code. It does not execute it.
    config.emit_terminal = [](oaknut::CodeGenerator&, EmitContext&) {};

    std::array<u32, 4> patterns{};
    oaknut::CodeGenerator pattern_code{patterns.data()};
    pattern_code.STR(X0, SP, 0);
    pattern_code.LDR(X0, SP, 0);
    pattern_code.STR(Q0, SP, 0);
    pattern_code.LDR(Q0, SP, 0);
    constexpr u32 operand_mask = 31 | (4095 << 10);
    for (int sample = 0; sample < 20; ++sample) {
        auto block = address_space.GenerateIR(IR::LocationDescriptor{0});
        std::array<u32, 16384> instructions{};
        oaknut::CodeGenerator code{instructions.data()};
        const auto result = EmitArm64(code, std::move(block), config, fastmem);
        size_t loads = 0, stores = 0;
        for (size_t i = 0; i < result.size / sizeof(u32); ++i) {
            for (size_t kind = 0; kind < patterns.size(); ++kind) {
                const auto word = instructions[i];
                if ((word & ~operand_mask) != patterns[kind])
                    continue;
                const size_t offset = ((word >> 10) & 4095) * (kind < 2 ? 8 : 16);
                if (offset < offsetof(StackLayout, spill) || offset >= offsetof(StackLayout, spill) + sizeof(StackLayout::spill))
                    continue;
                if (kind % 2)
                    ++loads;
                else
                    ++stores;
            }
        }
        std::printf("AUDIT_CODE workload=%s sample=%d words=%zu spill_loads=%zu spill_stores=%zu\n",
                    workload, sample, result.size / sizeof(u32), loads, stores);
    }
}
#endif
}  // namespace

TEST_CASE("A64: GPR and SIMD pressure preserves guest state", "[a64][spill-next-use]") {
    for (const auto& workload : workloads) {
        CAPTURE(workload.name);
        A64TestEnv env;
        PressureProgram program{env, workload};
        A64::Jit jit{A64::UserConfig{&env}};
        for (int iteration = 0; iteration < 8; ++iteration) {
            jit.ClearCache();
            program.Reset(jit, env);
            jit.Run();
            program.Check(jit);
        }
    }
}

// Opt-in microbenchmark. Report cold translation+one execution separately from
// warm execution. This is synthetic backend evidence, not an emulator FPS test.
TEST_CASE("A64: spill policy benchmark", "[.][spill-benchmark]") {
    for (const auto& workload : workloads) {
        CAPTURE(workload.name);
        A64TestEnv env;
        PressureProgram program{env, workload};
        A64::Jit jit{A64::UserConfig{&env}};
        for (int sample = 0; sample < 9; ++sample) {
            for (bool cold : {true, false}) {
                const int iterations = cold ? 100 : 10000;
                std::chrono::nanoseconds duration{};
                for (int i = 0; i < iterations; ++i) {
                    if (cold)
                        jit.ClearCache();
                    program.Reset(jit, env);
                    const auto begin = std::chrono::steady_clock::now();
                    jit.Run();
                    duration += std::chrono::steady_clock::now() - begin;
                }
                program.Check(jit);
                std::printf("AUDIT workload=%s phase=%s sample=%d ns_per_block=%.1f\n",
                            workload.name, cold ? "cold" : "warm", sample, double(duration.count()) / iterations);
            }
        }
#ifdef MCL_ARCHITECTURE_ARM64
        ReportCodeSize(env, workload.name);
#endif
    }
}
