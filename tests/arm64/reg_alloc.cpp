/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#include <catch2/catch_test_macros.hpp>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/ir/opcodes.h"

using namespace Dynarmic;
using namespace Dynarmic::Backend::Arm64;
using namespace oaknut::util;

TEST_CASE("ARM64: SIMD read-write move elimination", "[arm64][simd-copy]") {
    std::array<u32, 64> instructions{};
    oaknut::CodeGenerator code{instructions.data()};
    FpsrManager fpsr{code, 0};
    RegAlloc alloc{code, fpsr, {19, 20}, {8, 9, 10}};
    IR::Inst source{IR::Opcode::VectorNot};
    IR::Inst result{IR::Opcode::VectorNot};
    IR::Inst future{IR::Opcode::VectorAnd};
    IR::Inst alias{IR::Opcode::Identity};
    result.SetArg(0, IR::Value{&source});
    future.SetArg(0, IR::Value{&result});

    bool reuse = true;
    SECTION("Final use reuses the register without emitting a move") {}
    SECTION("A later source use requires a copy") {
        future.SetArg(0, IR::Value{&source});
        reuse = false;
    }
    SECTION("A live alias requires a copy") {
        alias.SetArg(0, IR::Value{&source});
        future.SetArg(0, IR::Value{&alias});
        reuse = false;
    }
    future.SetArg(1, IR::Value{&result});
    alloc.DefineAsRegister(&source, Q8);
    if (alias.UseCount()) {
        auto args = alloc.GetArgumentInfo(&alias);
        alloc.DefineAsExisting(&alias, args[0]);
        alloc.UpdateAllUses();
    }
    {
        auto args = alloc.GetArgumentInfo(&result);
        auto output = alloc.ReadWriteQ(args[0], &result);
        RegAlloc::Realize(output);
        REQUIRE((output->index() == 8) == reuse);
        REQUIRE(code.offset() == (reuse ? 0 : 4));
    }
    alloc.AssertAllUnlocked();
    alloc.UpdateAllUses();
    // The resulting value remains addressable after the old identity unlocks.
    {
        auto args = alloc.GetArgumentInfo(&future);
        auto output = alloc.ReadQ(args[1]);
        RegAlloc::Realize(output);
        REQUIRE((output->index() == 8) == reuse);
    }
    alloc.UpdateAllUses();
    alloc.AssertAllUnlocked();
    alloc.AssertNoMoreUses();
}

TEST_CASE("ARM64: repeated SIMD operands keep the source intact", "[arm64][simd-copy]") {
    std::array<u32, 64> instructions{};
    oaknut::CodeGenerator code{instructions.data()};
    FpsrManager fpsr{code, 0};
    RegAlloc alloc{code, fpsr, {19, 20}, {8, 9, 10}};
    IR::Inst source{IR::Opcode::VectorNot};
    IR::Inst result{IR::Opcode::VectorAnd};
    result.SetArg(0, IR::Value{&source});
    result.SetArg(1, IR::Value{&source});
    alloc.DefineAsRegister(&source, Q8);
    {
        auto args = alloc.GetArgumentInfo(&result);
        auto output = alloc.ReadWriteQ(args[0], &result);
        auto input = alloc.ReadQ(args[1]);
        RegAlloc::Realize(output, input);
        REQUIRE(output->index() != input->index());
        REQUIRE(input->index() == 8);
        REQUIRE(code.offset() == 4);
    }
    alloc.UpdateAllUses();
    alloc.AssertAllUnlocked();
    alloc.AssertNoMoreUses();
}
