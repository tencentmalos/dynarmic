/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/reg_alloc.h"

#include <catch2/catch_test_macros.hpp>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"

using namespace Dynarmic;
using namespace Dynarmic::Backend::Arm64;
using namespace oaknut::util;

TEST_CASE("ARM64: SIMD read-write move elimination", "[arm64][simd-copy]") {
    std::array<u32, 64> instructions{};
    oaknut::CodeGenerator code{instructions.data()};
    FpsrManager fpsr{code, 0};
    RegAlloc alloc{code, fpsr, {19, 20}, {8, 9, 10}, nullptr, true};
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
    RegAlloc alloc{code, fpsr, {19, 20}, {8, 9, 10}, nullptr, true};
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

TEST_CASE("ARM64: spills preserve the nearest future alias and locked values", "[arm64][spill-next-use]") {
    for (const bool vector : {false, true}) {
        std::array<u32, 256> instructions{};
        oaknut::CodeGenerator code{instructions.data()};
        FpsrManager fpsr{code, 0};
        IR::Inst near{vector ? IR::Opcode::VectorNot : IR::Opcode::Not64};
        IR::Inst far{vector ? IR::Opcode::VectorNot : IR::Opcode::Not64};
        IR::Inst result{vector ? IR::Opcode::VectorNot : IR::Opcode::Not64};
        IR::Inst alias{IR::Opcode::Identity};
        IR::Inst hold{IR::Opcode::Identity};
        IR::Block block{IR::LocationDescriptor{0}};
        bool use_alias = false, lock_far = false;
        SECTION("Nearest direct use stays in its register") {}
        SECTION("Nearest alias use stays in its register") {
            use_alias = true;
        }
        SECTION("A locked far value is never evicted") {
            lock_far = true;
        }
        if (use_alias)
            alias.SetArg(0, IR::Value{&near});
        if (lock_far)
            hold.SetArg(0, IR::Value{&far});
        block.AppendNewInst(IR::Opcode::Identity, {IR::Value{use_alias ? &alias : &near}});
        block.AppendNewInst(IR::Opcode::Identity, {IR::Value{&far}});
        RegAlloc alloc{code, fpsr, {19, 20}, {8, 9}, &block, true};
        alloc.DefineAsRegister(&near, vector ? oaknut::Reg{Q8} : oaknut::Reg{X19});
        alloc.DefineAsRegister(&far, vector ? oaknut::Reg{Q9} : oaknut::Reg{X20});
        if (use_alias) {
            auto args = alloc.GetArgumentInfo(&alias);
            alloc.DefineAsExisting(&alias, args[0]);
            alloc.UpdateAllUses();
        }
        const auto allocate = [&] {
            if (vector) {
                auto output = alloc.WriteQ(&result);
                RegAlloc::Realize(output);
                REQUIRE(output->index() == (lock_far ? 8 : 9));
            } else {
                auto output = alloc.WriteX(&result);
                RegAlloc::Realize(output);
                REQUIRE(output->index() == (lock_far ? 19 : 20));
            }
        };
        if (lock_far) {
            auto args = alloc.GetArgumentInfo(&hold);
            if (vector) {
                auto locked = alloc.ReadQ(args[0]);
                allocate();
            } else {
                auto locked = alloc.ReadX(args[0]);
                allocate();
            }
        } else {
            allocate();
        }
        REQUIRE(code.offset() == 4);  // Exactly one spill store.
        alloc.UpdateAllUses();
        size_t index = 0;
        for (auto& inst : block) {
            alloc.SetInstructionIndex(++index);
            {
                auto args = alloc.GetArgumentInfo(&inst);
                if (vector) {
                    auto input = alloc.ReadQ(args[0]);
                    RegAlloc::Realize(input);
                } else {
                    auto input = alloc.ReadX(args[0]);
                    RegAlloc::Realize(input);
                }
            }
            if (index == 1)
                REQUIRE(code.offset() == (lock_far ? 8 : 4));
            alloc.UpdateAllUses();
            alloc.AssertAllUnlocked();
        }
        REQUIRE(code.offset() == 8);  // Only the evicted value needs reloading.
        alloc.AssertNoMoreUses();
    }
}

TEST_CASE("ARM64: table operands stay live until their lookup", "[arm64][spill-next-use]") {
    std::array<u32, 64> instructions{};
    oaknut::CodeGenerator code{instructions.data()};
    FpsrManager fpsr{code, 0};
    IR::Inst near{IR::Opcode::VectorNot};
    IR::Inst far{IR::Opcode::VectorNot};
    IR::Inst result{IR::Opcode::VectorNot};
    IR::Inst zero{IR::Opcode::VectorBroadcast64};
    zero.SetArg(0, IR::Value{u64{0}});
    IR::Block block{IR::LocationDescriptor{0}};
    block.AppendNewInst(IR::Opcode::VectorTable, {IR::Value{&near}, {}, {}, {}});
    auto* table = &block.back();
    block.AppendNewInst(IR::Opcode::Void, {});
    block.AppendNewInst(IR::Opcode::VectorTableLookup128, {IR::Value{&zero}, IR::Value{table}, IR::Value{&zero}});
    block.AppendNewInst(IR::Opcode::Identity, {IR::Value{&far}});
    auto* far_use = &block.back();
    RegAlloc alloc{code, fpsr, {19, 20}, {8, 9}, &block, true};
    alloc.DefineAsRegister(&near, Q8);
    alloc.DefineAsRegister(&far, Q9);
    alloc.SetInstructionIndex(2);
    {
        auto output = alloc.WriteQ(&result);
        RegAlloc::Realize(output);
        REQUIRE(output->index() == 9);
    }
    alloc.UpdateAllUses();
    alloc.SetInstructionIndex(3);
    {
        auto args = alloc.GetArgumentInfo(table);
        auto input = alloc.ReadQ(args[0]);
        RegAlloc::Realize(input);
        REQUIRE(input->index() == 8);
        REQUIRE(code.offset() == 4);
    }
    alloc.UpdateAllUses();
    alloc.SetInstructionIndex(4);
    {
        auto args = alloc.GetArgumentInfo(far_use);
        auto input = alloc.ReadQ(args[0]);
        RegAlloc::Realize(input);
        REQUIRE(code.offset() == 8);
    }
    alloc.UpdateAllUses();
    alloc.AssertAllUnlocked();
    alloc.AssertNoMoreUses();
}

TEST_CASE("ARM64: bit select chooses a dying mask or data operand", "[arm64][bit-select]") {
    for (int live_mask = 0; live_mask < 8; ++live_mask) {
        CAPTURE(live_mask);
        std::array<u32, 128> instructions{}, expected{};
        oaknut::CodeGenerator code{instructions.data()}, reference{expected.data()};
        FpsrManager fpsr{code, 0};
        IR::Block block{IR::LocationDescriptor{0}};
        RegAlloc alloc{code, fpsr, {19, 20}, {8, 9, 10, 11}, &block, true};
        IR::Inst mask{IR::Opcode::VectorNot}, yes{IR::Opcode::VectorNot}, no{IR::Opcode::VectorNot};
        IR::Inst select{IR::Opcode::VectorBitSelect};
        std::array<IR::Inst, 3> future{IR::Inst{IR::Opcode::Identity}, IR::Inst{IR::Opcode::Identity}, IR::Inst{IR::Opcode::Identity}};
        IR::Inst* inputs[]{&mask, &yes, &no};
        for (int i = 0; i < 3; ++i) {
            select.SetArg(i, IR::Value{inputs[i]});
            if (live_mask & (1 << i))
                future[i].SetArg(0, IR::Value{inputs[i]});
            alloc.DefineAsRegister(inputs[i], oaknut::QReg{8 + i});
        }
        EmitConfig config{};
        EmittedBlockInfo emitted{};
        Backend::ExceptionHandler exception_handler;
        FastmemManager fastmem{exception_handler};
        EmitContext ctx{block, alloc, config, emitted, fpsr, fastmem, {}};
        EmitIR<IR::Opcode::VectorBitSelect>(code, ctx, &select);
        if (!(live_mask & 1))
            reference.BSL(Q8.B16(), Q9.B16(), Q10.B16());
        else if (!(live_mask & 2))
            reference.BIF(Q9.B16(), Q10.B16(), Q8.B16());
        else if (!(live_mask & 4))
            reference.BIT(Q10.B16(), Q9.B16(), Q8.B16());
        else {
            reference.MOV(Q11.B16(), Q8.B16());
            reference.BSL(Q11.B16(), Q9.B16(), Q10.B16());
        }
        REQUIRE(code.offset() == reference.offset());
        REQUIRE(instructions == expected);
        alloc.AssertAllUnlocked();
        alloc.UpdateAllUses();
        for (int i = 0; i < 3; ++i) {
            if (!(live_mask & (1 << i)))
                continue;
            {
                auto args = alloc.GetArgumentInfo(&future[i]);
                auto input = alloc.ReadQ(args[0]);
                RegAlloc::Realize(input);
                REQUIRE(input->index() == 8 + i);
            }
            alloc.UpdateAllUses();
        }
        alloc.AssertNoMoreUses();
    }
}

TEST_CASE("ARM64: indexed aliases survive register classes, flags and calls", "[arm64][value-location]") {
    std::array<u32, 512> instructions{};
    oaknut::CodeGenerator code{instructions.data()};
    FpsrManager fpsr{code, 0};
    IR::Block block{IR::LocationDescriptor{0}};
    block.AppendNewInst(IR::Opcode::Not64, {IR::Value{u64{0}}});
    auto* source = &block.back();
    block.AppendNewInst(IR::Opcode::Identity, {IR::Value{source}});
    auto* alias = &block.back();
    block.AppendNewInst(IR::Opcode::Identity, {IR::Value{alias}});
    auto* use_gpr = &block.back();
    block.AppendNewInst(IR::Opcode::Identity, {IR::Value{source}});
    auto* use_fpr = &block.back();
    block.AppendNewInst(IR::Opcode::Identity, {IR::Value{alias}});
    auto* call = &block.back();
    block.AppendNewInst(IR::Opcode::Identity, {IR::Value{source}});
    auto* after_call = &block.back();
    block.AppendNewInst(IR::Opcode::Identity, {IR::Value{alias}});
    auto* last = &block.back();
    // Deliberately duplicate names before the final allocator indexing pass.
    for (auto& inst : block)
        inst.SetName(99);
    RegAlloc alloc{code, fpsr, {0, 1, 2}, {0, 1, 2}, &block, true};
    REQUIRE(source->GetName() == 1);
    REQUIRE(last->GetName() == block.size());
    alloc.DefineAsRegister(source, Q0);
    {
        auto args = alloc.GetArgumentInfo(alias);
        alloc.DefineAsExisting(alias, args[0]);
    }
    alloc.UpdateAllUses();
    {
        auto args = alloc.GetArgumentInfo(use_gpr);
        auto value = alloc.ReadX(args[0]);
        RegAlloc::Realize(value);
        REQUIRE(args[0].IsInGpr());
    }
    alloc.UpdateAllUses();
    {
        auto args = alloc.GetArgumentInfo(use_fpr);
        auto value = alloc.ReadD(args[0]);
        RegAlloc::Realize(value);
        REQUIRE(args[0].IsInFpr());
    }
    alloc.UpdateAllUses();
    {
        auto args = alloc.GetArgumentInfo(call);
        alloc.PrepareForCall(args[0]);
        REQUIRE(args[0].CurrentLocationKind() == HostLoc::Kind::Spill);
    }
    alloc.UpdateAllUses();
    {
        auto args = alloc.GetArgumentInfo(after_call);
        auto value = alloc.ReadX(args[0]);
        RegAlloc::Realize(value);
    }
    alloc.UpdateAllUses();
    alloc.PrepareForCall();
    {
        auto args = alloc.GetArgumentInfo(last);
        auto value = alloc.ReadD(args[0]);
        RegAlloc::Realize(value);
    }
    alloc.UpdateAllUses();
    alloc.AssertAllUnlocked();
    alloc.AssertNoMoreUses();

    IR::Inst flag_value{IR::Opcode::GetNZCVFromOp}, flag_alias{IR::Opcode::Identity}, flag_use{IR::Opcode::Identity};
    flag_alias.SetArg(0, IR::Value{&flag_value});
    flag_use.SetArg(0, IR::Value{&flag_alias});
    IR::Inst next_flags{IR::Opcode::GetNZCVFromOp}, next_use{IR::Opcode::Identity};
    next_use.SetArg(0, IR::Value{&next_flags});
    {
        auto output = alloc.WriteFlags(&flag_value);
        RegAlloc::Realize(output);
    }
    {
        auto args = alloc.GetArgumentInfo(&flag_alias);
        alloc.DefineAsExisting(&flag_alias, args[0]);
    }
    alloc.UpdateAllUses();
    alloc.SpillFlags();
    alloc.PrepareForCall();
    {
        auto args = alloc.GetArgumentInfo(&flag_use);
        REQUIRE(args[0].CurrentLocationKind() == HostLoc::Kind::Spill);
        alloc.ReadWriteFlags(args[0], &next_flags);
    }
    alloc.UpdateAllUses();
    {
        auto args = alloc.GetArgumentInfo(&next_use);
        REQUIRE(args[0].CurrentLocationKind() == HostLoc::Kind::Flags);
        alloc.ReadWriteFlags(args[0], nullptr);
    }
    alloc.UpdateAllUses();
    alloc.AssertAllUnlocked();
    alloc.AssertNoMoreUses();
}
