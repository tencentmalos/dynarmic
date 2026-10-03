/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/reg_alloc.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>

#include <mcl/assert.hpp>
#include <mcl/bit/bit_field.hpp>
#include <mcl/bit_cast.hpp>
#include <mcl/mp/metavalue/lift_value.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/verbose_debugging_output.h"
#include "dynarmic/common/always_false.h"
#include "dynarmic/ir/basic_block.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

constexpr size_t spill_offset = offsetof(StackLayout, spill);
constexpr size_t spill_slot_size = sizeof(decltype(StackLayout::spill)::value_type);

static bool IsValuelessType(IR::Type type) {
    switch (type) {
    case IR::Type::Table:
        return true;
    default:
        return false;
    }
}

IR::Type Argument::GetType() const {
    return value.GetType();
}

bool Argument::IsImmediate() const {
    return value.IsImmediate();
}

bool Argument::GetImmediateU1() const {
    return value.GetU1();
}

u8 Argument::GetImmediateU8() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100);
    return u8(imm);
}

u16 Argument::GetImmediateU16() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x10000);
    return u16(imm);
}

u32 Argument::GetImmediateU32() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100000000);
    return u32(imm);
}

u64 Argument::GetImmediateU64() const {
    return value.GetImmediateAsU64();
}

IR::Cond Argument::GetImmediateCond() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::Cond);
    return value.GetCond();
}

IR::AccType Argument::GetImmediateAccType() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::AccType);
    return value.GetAccType();
}

HostLoc::Kind Argument::CurrentLocationKind() const {
    return reg_alloc.ValueLocation(value.GetInst())->kind;
}

bool HostLocInfo::Contains(const IR::Inst* value) const {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void HostLocInfo::SetupScratchLocation() {
    ASSERT(IsCompletelyEmpty());
    realized = true;
}

void HostLocInfo::SetupLocation(const IR::Inst* value) {
    ASSERT(IsCompletelyEmpty());
    values.clear();
    values.push_back(value);
    realized = true;
    uses_this_inst = 0;
    accumulated_uses = 0;
    expected_uses = value->UseCount();
}

bool HostLocInfo::IsCompletelyEmpty() const {
    return values.empty() && !locked && !realized && !accumulated_uses && !expected_uses && !uses_this_inst;
}

bool HostLocInfo::MaybeAllocatable() const {
    return !locked && !realized;
}

bool HostLocInfo::IsOneRemainingUse() const {
    return accumulated_uses + 1 == expected_uses && uses_this_inst == 1;
}

void HostLocInfo::UpdateUses() {
    accumulated_uses += uses_this_inst;
    uses_this_inst = 0;

    if (accumulated_uses == expected_uses) {
        values.clear();
        accumulated_uses = 0;
        expected_uses = 0;
    }
}

RegAlloc::RegAlloc(oaknut::CodeGenerator& code, FpsrManager& fpsr_manager, std::vector<int> gpr_order, std::vector<int> fpr_order, IR::Block* block, bool verify_locations)
        : code{code}, fpsr_manager{fpsr_manager}, gpr_order{std::move(gpr_order)}, fpr_order{std::move(fpr_order)}, verify_locations{verify_locations}, block{block} {
#ifndef NDEBUG
    this->verify_locations = true;
#endif
    if (block) {
        value_locations.reserve(block->size());
        for (auto& inst : *block) {
            inst.SetName(static_cast<unsigned>(value_locations.size() + 1));
            value_locations.push_back({&inst, std::nullopt});
        }
    }
    touched_locations.reserve(16);
}

RegAlloc::ArgumentInfo RegAlloc::GetArgumentInfo(IR::Inst* inst) {
    ArgumentInfo ret = {Argument{*this}, Argument{*this}, Argument{*this}, Argument{*this}};
    for (size_t i = 0; i < inst->NumArgs(); i++) {
        const IR::Value arg = inst->GetArg(i);
        ret[i].value = arg;
        if (!arg.IsImmediate() && !IsValuelessType(arg.GetType())) {
            const auto location = ValueLocation(arg.GetInst());
            ASSERT_MSG(location, "argument must already been defined");
            ValueInfo(*location).uses_this_inst++;
            TouchLocation(*location);
        }
    }
    return ret;
}

bool RegAlloc::WasValueDefined(IR::Inst* inst) const {
    return defined_insts.count(inst) > 0;
}

bool RegAlloc::CanReuseFpr(const Argument& arg) const {
    if (arg.IsImmediate()) {
        return false;
    }
    const auto loc = ValueLocation(arg.value.GetInst());
    if (!loc || loc->kind != HostLoc::Kind::Fpr) {
        return false;
    }
    const auto& info = fprs[loc->index];
    return info.IsOneRemainingUse() && !info.locked && !info.realized;
}

void RegAlloc::PrepareForCall(std::optional<Argument::copyable_reference> arg0, std::optional<Argument::copyable_reference> arg1, std::optional<Argument::copyable_reference> arg2, std::optional<Argument::copyable_reference> arg3) {
    fpsr_manager.Spill();
    SpillFlags();

    // TODO: Spill into callee-save registers

    for (int i = 0; i < 32; i++) {
        if (mcl::bit::get_bit(i, static_cast<u32>(ABI_CALLER_SAVE))) {
            SpillGpr(i);
        }
    }

    for (int i = 0; i < 32; i++) {
        if (mcl::bit::get_bit(i, static_cast<u32>(ABI_CALLER_SAVE >> 32))) {
            SpillFpr(i);
        }
    }

    const std::array<std::optional<Argument::copyable_reference>, 4> args{arg0, arg1, arg2, arg3};

    // AAPCS64 Next General-purpose Register Number
    int ngrn = 0;
    // AAPCS64 Next SIMD and Floating-point Register Number
    int nsrn = 0;

    for (int i = 0; i < 4; i++) {
        if (args[i]) {
            if (args[i]->get().GetType() == IR::Type::U128) {
                ASSERT(fprs[nsrn].IsCompletelyEmpty());
                LoadCopyInto(args[i]->get().value, oaknut::QReg{nsrn});
                nsrn++;
            } else {
                ASSERT(gprs[ngrn].IsCompletelyEmpty());
                LoadCopyInto(args[i]->get().value, oaknut::XReg{ngrn});
                ngrn++;
            }
        } else {
            // Gaps are assumed to be in general-purpose registers
            // TODO: should there be a separate list passed for FPRs instead?
            ngrn++;
        }
    }
}

void RegAlloc::DefineAsExisting(IR::Inst* inst, Argument& arg) {
    defined_insts.insert(inst);

    ASSERT(!ValueLocation(inst));

    if (arg.value.IsImmediate()) {
        inst->ReplaceUsesWith(arg.value);
        return;
    }

    const auto location = ValueLocation(arg.value.GetInst());
    ASSERT(location);
    auto& info = ValueInfo(*location);
    info.values.push_back(inst);
    info.expected_uses += inst->UseCount();
    SetValueLocation(inst, location);
    TouchLocation(*location);
}

void RegAlloc::DefineAsRegister(IR::Inst* inst, oaknut::Reg reg) {
    defined_insts.insert(inst);

    ASSERT(!ValueLocation(inst));
    const HostLoc location{reg.is_vector() ? HostLoc::Kind::Fpr : HostLoc::Kind::Gpr, reg.index()};
    SetupLocation(location, inst);
    ValueInfo(location).realized = false;
}

void RegAlloc::UpdateAllUses() {
    for (const auto location : touched_locations) {
        auto& info = ValueInfo(location);
        if (info.accumulated_uses + info.uses_this_inst == info.expected_uses) {
            for (const auto* value : info.values) {
                SetValueLocation(value, std::nullopt);
            }
        }
        info.UpdateUses();
    }
    touched_locations.clear();
    touched_mask.reset();
    if (verify_locations) {
        const auto verify = [&](const HostLocInfo& info) {
            ASSERT(info.uses_this_inst == 0);
            ASSERT(info.values.empty() || info.accumulated_uses < info.expected_uses);
            for (const auto* value : info.values) {
                ASSERT(ValueLocation(value) == ScanValueLocation(value));
            }
        };
        for (const auto& info : gprs)
            verify(info);
        for (const auto& info : fprs)
            verify(info);
        verify(flags);
        for (const auto& info : spills)
            verify(info);
        for (const auto& entry : value_locations) {
            ASSERT(entry.location == ScanValueLocation(entry.value));
        }
        for (const auto& [value, location] : external_locations) {
            ASSERT(location == ScanValueLocation(value));
        }
    }
}

void RegAlloc::AssertAllUnlocked() const {
    const auto is_unlocked = [](const auto& i) { return !i.locked && !i.realized; };
    ASSERT(std::all_of(gprs.begin(), gprs.end(), is_unlocked));
    ASSERT(std::all_of(fprs.begin(), fprs.end(), is_unlocked));
    ASSERT(is_unlocked(flags));
    ASSERT(std::all_of(spills.begin(), spills.end(), is_unlocked));
}

void RegAlloc::AssertNoMoreUses() const {
    const auto is_empty = [](const auto& i) { return i.IsCompletelyEmpty(); };
    ASSERT(std::all_of(gprs.begin(), gprs.end(), is_empty));
    ASSERT(std::all_of(fprs.begin(), fprs.end(), is_empty));
    ASSERT(is_empty(flags));
    ASSERT(std::all_of(spills.begin(), spills.end(), is_empty));
}

void RegAlloc::EmitVerboseDebuggingOutput() {
    code.MOV(X19, mcl::bit_cast<u64>(&PrintVerboseDebuggingOutputLine));  // Non-volatile register

    const auto do_location = [&](HostLocInfo& info, HostLocType type, size_t index) {
        using namespace oaknut::util;
        for (const IR::Inst* value : info.values) {
            code.MOV(X0, SP);
            code.MOV(X1, static_cast<u64>(type));
            code.MOV(X2, index);
            code.MOV(X3, value->GetName());
            code.MOV(X4, static_cast<u64>(value->GetType()));
            code.BLR(X19);
        }
    };

    for (size_t i = 0; i < gprs.size(); i++) {
        do_location(gprs[i], HostLocType::X, i);
    }
    for (size_t i = 0; i < fprs.size(); i++) {
        do_location(fprs[i], HostLocType::Q, i);
    }
    do_location(flags, HostLocType::Nzcv, 0);
    for (size_t i = 0; i < spills.size(); i++) {
        do_location(spills[i], HostLocType::Spill, i);
    }
}

template<HostLoc::Kind kind>
int RegAlloc::GenerateImmediate(const IR::Value& value) {
    ASSERT(value.GetType() != IR::Type::U1);
    if constexpr (kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order);
        SpillGpr(new_location_index);
        gprs[new_location_index].SetupScratchLocation();

        code.MOV(oaknut::XReg{new_location_index}, value.GetImmediateAsU64());

        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order);
        SpillFpr(new_location_index);
        fprs[new_location_index].SetupScratchLocation();

        code.MOV(Xscratch0, value.GetImmediateAsU64());
        code.FMOV(oaknut::DReg{new_location_index}, Xscratch0);

        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        SpillFlags();
        flags.SetupScratchLocation();

        code.MOV(Xscratch0, value.GetImmediateAsU64());
        code.MSR(oaknut::SystemReg::NZCV, Xscratch0);

        return 0;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<kind>>);
    }
}

template<HostLoc::Kind required_kind>
int RegAlloc::RealizeReadImpl(const IR::Value& value) {
    if (value.IsImmediate()) {
        return GenerateImmediate<required_kind>(value);
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);

    if (current_location->kind == required_kind) {
        ValueInfo(*current_location).realized = true;
        return current_location->index;
    }

    ASSERT(!ValueInfo(*current_location).realized);
    ASSERT(ValueInfo(*current_location).locked);

    if constexpr (required_kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order);
        SpillGpr(new_location_index);

        switch (current_location->kind) {
        case HostLoc::Kind::Gpr:
            ASSERT_FALSE("Logic error");
            break;
        case HostLoc::Kind::Fpr:
            code.FMOV(oaknut::XReg{new_location_index}, oaknut::DReg{current_location->index});
            // ASSERT size fits
            break;
        case HostLoc::Kind::Spill:
            code.LDR(oaknut::XReg{new_location_index}, SP, spill_offset + current_location->index * spill_slot_size);
            break;
        case HostLoc::Kind::Flags:
            code.MRS(oaknut::XReg{new_location_index}, oaknut::SystemReg::NZCV);
            break;
        }

        MoveLocation(*current_location, {HostLoc::Kind::Gpr, new_location_index});
        gprs[new_location_index].realized = true;
        return new_location_index;
    } else if constexpr (required_kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order);
        SpillFpr(new_location_index);

        switch (current_location->kind) {
        case HostLoc::Kind::Gpr:
            code.FMOV(oaknut::DReg{new_location_index}, oaknut::XReg{current_location->index});
            break;
        case HostLoc::Kind::Fpr:
            ASSERT_FALSE("Logic error");
            break;
        case HostLoc::Kind::Spill:
            code.LDR(oaknut::QReg{new_location_index}, SP, spill_offset + current_location->index * spill_slot_size);
            break;
        case HostLoc::Kind::Flags:
            ASSERT_FALSE("Moving from flags into fprs is not currently supported");
            break;
        }

        MoveLocation(*current_location, {HostLoc::Kind::Fpr, new_location_index});
        fprs[new_location_index].realized = true;
        return new_location_index;
    } else if constexpr (required_kind == HostLoc::Kind::Flags) {
        ASSERT_FALSE("A simple read from flags is likely a logic error.");
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<required_kind>>);
    }
}

template<HostLoc::Kind kind>
int RegAlloc::RealizeWriteImpl(const IR::Inst* value) {
    defined_insts.insert(value);

    ASSERT(!ValueLocation(value));

    if constexpr (kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order);
        SpillGpr(new_location_index);
        SetupLocation({HostLoc::Kind::Gpr, new_location_index}, value);
        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order);
        SpillFpr(new_location_index);
        SetupLocation({HostLoc::Kind::Fpr, new_location_index}, value);
        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        SpillFlags();
        SetupLocation({HostLoc::Kind::Flags, 0}, value);
        return 0;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<kind>>);
    }
}

template<HostLoc::Kind kind>
int RegAlloc::RealizeReadWriteImpl(const IR::Value& read_value, const IR::Inst* write_value) {
    defined_insts.insert(write_value);

    // A destructive SIMD operand can keep its register at its final use.
    // Account for all aliases, and retain the old identity until RAReg unlocks
    // it. Multiple operands using the same value must still receive a copy.
    if constexpr (kind == HostLoc::Kind::Fpr) {
        if (!read_value.IsImmediate()) {
            const auto location = ValueLocation(read_value.GetInst());
            if (location && location->kind == kind) {
                auto& info = ValueInfo(*location);
                if (info.IsOneRemainingUse() && info.locked == 1 && !info.realized) {
                    info.values.push_back(write_value);
                    info.expected_uses += write_value->UseCount();
                    SetValueLocation(write_value, location);
                    TouchLocation(*location);
                    info.realized = true;
                    return location->index;
                }
            }
        }
    }

    const int write_loc = RealizeWriteImpl<kind>(write_value);

    if constexpr (kind == HostLoc::Kind::Gpr) {
        LoadCopyInto(read_value, oaknut::XReg{write_loc});
        return write_loc;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        LoadCopyInto(read_value, oaknut::QReg{write_loc});
        return write_loc;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        ASSERT_FALSE("Incorrect function for ReadWrite of flags");
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<kind>>);
    }
}

template int RegAlloc::RealizeReadImpl<HostLoc::Kind::Gpr>(const IR::Value& value);
template int RegAlloc::RealizeReadImpl<HostLoc::Kind::Fpr>(const IR::Value& value);
template int RegAlloc::RealizeReadImpl<HostLoc::Kind::Flags>(const IR::Value& value);
template int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Gpr>(const IR::Inst* value);
template int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Fpr>(const IR::Inst* value);
template int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Flags>(const IR::Inst* value);
template int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Gpr>(const IR::Value&, const IR::Inst*);
template int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Fpr>(const IR::Value&, const IR::Inst*);
template int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Flags>(const IR::Value&, const IR::Inst*);

int RegAlloc::AllocateRegister(const std::array<HostLocInfo, 32>& regs, const std::vector<int>& order) const {
    const auto empty = std::find_if(order.begin(), order.end(), [&](int i) { return regs[i].IsCompletelyEmpty(); });
    if (empty != order.end()) {
        return *empty;
    }

    if (block && !future_uses_ready) {
        size_t index = 0;
        for (const auto& inst : *block) {
            ++index;
            // VectorTable keeps its operands live until the lookup emitter
            // consumes them, rather than consuming them at the table node.
            if (IsValuelessType(inst.GetType())) {
                continue;
            }
            const auto record_use = [&](const IR::Value& arg) {
                if (!arg.IsImmediate()) {
                    future_uses[arg.GetInst()].push_back(index);
                }
            };
            for (size_t arg_index = 0; arg_index < inst.NumArgs(); ++arg_index) {
                const auto arg = inst.GetArg(arg_index);
                if (IsValuelessType(arg.GetType())) {
                    const auto& table = *arg.GetInst();
                    for (size_t i = 0; i < table.NumArgs(); ++i) {
                        record_use(table.GetArg(i));
                    }
                } else {
                    record_use(arg);
                }
            }
        }
        future_uses_ready = true;
    }

    int candidate = -1;
    size_t furthest = 0;
    for (const int i : order) {
        if (!regs[i].MaybeAllocatable()) {
            continue;
        }
        size_t next_use = std::numeric_limits<size_t>::max();
        // A register can contain several IR aliases. Its nearest remaining
        // alias use determines how soon a spill would need to be reloaded.
        for (const auto* value : regs[i].values) {
            const auto uses = future_uses.find(value);
            if (uses == future_uses.end()) {
                continue;
            }
            const auto next = std::upper_bound(uses->second.begin(), uses->second.end(), instruction_index);
            if (next != uses->second.end()) {
                next_use = std::min(next_use, *next);
            }
        }
        if (candidate == -1 || next_use > furthest) {
            candidate = i;
            furthest = next_use;
        }
    }
    ASSERT_MSG(candidate != -1, "All registers are locked or realized");
    return candidate;
}

void RegAlloc::SpillGpr(int index) {
    ASSERT(!gprs[index].locked && !gprs[index].realized);
    if (gprs[index].values.empty()) {
        return;
    }
    const int new_location_index = FindFreeSpill();
    code.STR(oaknut::XReg{index}, SP, spill_offset + new_location_index * spill_slot_size);
    MoveLocation({HostLoc::Kind::Gpr, index}, {HostLoc::Kind::Spill, new_location_index});
}

void RegAlloc::SpillFpr(int index) {
    ASSERT(!fprs[index].locked && !fprs[index].realized);
    if (fprs[index].values.empty()) {
        return;
    }
    const int new_location_index = FindFreeSpill();
    code.STR(oaknut::QReg{index}, SP, spill_offset + new_location_index * spill_slot_size);
    MoveLocation({HostLoc::Kind::Fpr, index}, {HostLoc::Kind::Spill, new_location_index});
}

void RegAlloc::ReadWriteFlags(Argument& read, IR::Inst* write) {
    defined_insts.insert(write);

    const auto current_location = ValueLocation(read.value.GetInst());
    ASSERT(current_location);

    if (current_location->kind == HostLoc::Kind::Flags) {
        if (!flags.IsOneRemainingUse()) {
            SpillFlags();
        }
    } else if (current_location->kind == HostLoc::Kind::Gpr) {
        if (!flags.values.empty()) {
            SpillFlags();
        }
        code.MSR(oaknut::SystemReg::NZCV, oaknut::XReg{current_location->index});
    } else if (current_location->kind == HostLoc::Kind::Spill) {
        if (!flags.values.empty()) {
            SpillFlags();
        }
        code.LDR(Wscratch0, SP, spill_offset + current_location->index * spill_slot_size);
        code.MSR(oaknut::SystemReg::NZCV, Xscratch0);
    } else {
        ASSERT_FALSE("Invalid current location for flags");
    }

    if (write) {
        SetupLocation({HostLoc::Kind::Flags, 0}, write);
        flags.realized = false;
    }
}

void RegAlloc::SpillFlags() {
    ASSERT(!flags.locked && !flags.realized);
    if (flags.values.empty()) {
        return;
    }
    const int new_location_index = AllocateRegister(gprs, gpr_order);
    SpillGpr(new_location_index);
    code.MRS(oaknut::XReg{new_location_index}, oaknut::SystemReg::NZCV);
    MoveLocation({HostLoc::Kind::Flags, 0}, {HostLoc::Kind::Gpr, new_location_index});
}

int RegAlloc::FindFreeSpill() const {
    const auto iter = std::find_if(spills.begin(), spills.end(), [](const HostLocInfo& info) { return info.values.empty(); });
    ASSERT_MSG(iter != spills.end(), "All spill locations are full");
    return static_cast<int>(iter - spills.begin());
}

void RegAlloc::LoadCopyInto(const IR::Value& value, oaknut::XReg reg) {
    if (value.IsImmediate()) {
        code.MOV(reg, value.GetImmediateAsU64());
        return;
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);
    switch (current_location->kind) {
    case HostLoc::Kind::Gpr:
        code.MOV(reg, oaknut::XReg{current_location->index});
        break;
    case HostLoc::Kind::Fpr:
        code.FMOV(reg, oaknut::DReg{current_location->index});
        // ASSERT size fits
        break;
    case HostLoc::Kind::Spill:
        code.LDR(reg, SP, spill_offset + current_location->index * spill_slot_size);
        break;
    case HostLoc::Kind::Flags:
        code.MRS(reg, oaknut::SystemReg::NZCV);
        break;
    }
}

void RegAlloc::LoadCopyInto(const IR::Value& value, oaknut::QReg reg) {
    if (value.IsImmediate()) {
        code.MOV(Xscratch0, value.GetImmediateAsU64());
        code.FMOV(reg.toD(), Xscratch0);
        return;
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);
    switch (current_location->kind) {
    case HostLoc::Kind::Gpr:
        code.FMOV(reg.toD(), oaknut::XReg{current_location->index});
        break;
    case HostLoc::Kind::Fpr:
        code.MOV(reg.B16(), oaknut::QReg{current_location->index}.B16());
        break;
    case HostLoc::Kind::Spill:
        // TODO: Minimize move size to max value width
        code.LDR(reg, SP, spill_offset + current_location->index * spill_slot_size);
        break;
    case HostLoc::Kind::Flags:
        ASSERT_FALSE("Moving from flags into fprs is not currently supported");
        break;
    }
}

std::optional<HostLoc> RegAlloc::ScanValueLocation(const IR::Inst* value) const {
    const auto contains_value = [value](const HostLocInfo& info) { return info.Contains(value); };

    if (const auto iter = std::find_if(gprs.begin(), gprs.end(), contains_value); iter != gprs.end()) {
        return HostLoc{HostLoc::Kind::Gpr, static_cast<int>(iter - gprs.begin())};
    }
    if (const auto iter = std::find_if(fprs.begin(), fprs.end(), contains_value); iter != fprs.end()) {
        return HostLoc{HostLoc::Kind::Fpr, static_cast<int>(iter - fprs.begin())};
    }
    if (contains_value(flags)) {
        return HostLoc{HostLoc::Kind::Flags, 0};
    }
    if (const auto iter = std::find_if(spills.begin(), spills.end(), contains_value); iter != spills.end()) {
        return HostLoc{HostLoc::Kind::Spill, static_cast<int>(iter - spills.begin())};
    }
    return std::nullopt;
}

HostLocInfo& RegAlloc::ValueInfo(HostLoc host_loc) {
    switch (host_loc.kind) {
    case HostLoc::Kind::Gpr:
        return gprs[static_cast<size_t>(host_loc.index)];
    case HostLoc::Kind::Fpr:
        return fprs[static_cast<size_t>(host_loc.index)];
    case HostLoc::Kind::Flags:
        return flags;
    case HostLoc::Kind::Spill:
        return spills[static_cast<size_t>(host_loc.index)];
    }
    ASSERT_FALSE("RegAlloc::ValueInfo: Invalid HostLoc::Kind");
}

HostLocInfo& RegAlloc::ValueInfo(const IR::Inst* value) {
    const auto location = ValueLocation(value);
    ASSERT_MSG(location, "RegAlloc::ValueInfo: Value not found");
    return ValueInfo(*location);
}

std::optional<HostLoc> RegAlloc::ValueLocation(const IR::Inst* value) const {
    std::optional<HostLoc> location;
    // IR reserves name zero for unnamed instructions. Unsigned underflow
    // takes such external values directly to the pointer-keyed fallback.
    const auto name = value->GetName() - 1;
    if (name < value_locations.size() && value_locations[name].value == value) {
        location = value_locations[name].location;
    } else if (const auto it = external_locations.find(value); it != external_locations.end()) {
        location = it->second;
    }
    if (verify_locations) {
        ASSERT(location == ScanValueLocation(value));
    }
    return location;
}

void RegAlloc::SetValueLocation(const IR::Inst* value, std::optional<HostLoc> location) {
    // IR reserves name zero for unnamed instructions. Unsigned underflow
    // takes such external values directly to the pointer-keyed fallback.
    const auto name = value->GetName() - 1;
    if (name < value_locations.size() && value_locations[name].value == value) {
        value_locations[name].location = location;
    } else if (location) {
        external_locations.insert_or_assign(value, *location);
    } else {
        external_locations.erase(value);
    }
}

void RegAlloc::TouchLocation(HostLoc location) {
    const auto index = location.kind == HostLoc::Kind::Gpr   ? location.index
                     : location.kind == HostLoc::Kind::Fpr   ? 32 + location.index
                     : location.kind == HostLoc::Kind::Flags ? 64
                                                             : 65 + location.index;
    if (!touched_mask.test(index)) {
        touched_mask.set(index);
        touched_locations.push_back(location);
    }
}

void RegAlloc::SetupLocation(HostLoc location, const IR::Inst* value) {
    ValueInfo(location).SetupLocation(value);
    SetValueLocation(value, location);
    TouchLocation(location);
}

void RegAlloc::MoveLocation(HostLoc from, HostLoc to) {
    ASSERT(ValueInfo(to).IsCompletelyEmpty());
    auto& destination = ValueInfo(to);
    destination = std::exchange(ValueInfo(from), {});
    for (const auto* value : destination.values) {
        SetValueLocation(value, to);
    }
    TouchLocation(from);
    TouchLocation(to);
}

}  // namespace Dynarmic::Backend::Arm64
