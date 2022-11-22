const std = @import("std");
const expect = std.testing.expect;
const qv = @import("QVec.zig");
const header = @cImport({
    @cDefine("ZIG_IMPORT", "1");
    @cInclude("Header.h");
});

pub fn ComputeUnit(comptime T: type, comptime qVecSize: comptime_int, comptime tNi: type, comptime memoryLayout: header.enum_MemoryLayoutType, comptime streaming: header.enum_Streaming) type {
    return struct {
        const FieldAccessType = qv.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming);
        const FlowParamsType = qv.FlowParams(T);
        const ForceType = qv.Force(T);
        fa: FieldAccessType,
        flow: FlowParamsType,
        F: [*]ForceType,
        Nu: [*]T,
    };
}
