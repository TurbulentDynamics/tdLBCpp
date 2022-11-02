const std = @import("std");
const expect = std.testing.expect;
const header = @cImport({
    @cDefine("ZIG_IMPORT", "1");
    @cInclude("Header.h");
});

pub fn Vec3(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,
        z: T,
    };
}

test "vec3 test" {
    var a = Vec3(f32){
        .x = 1.0,
        .y = 2.0,
        .z = 3.0,
    };
    try expect(a.x == 1.0);
    try expect(a.y == 2.0);
    try expect(a.z == 3.0);
}

pub fn Velocity(comptime T: type) type {
    return Vec3(T);
}

pub fn Force(comptime T: type) type {
    return Vec3(T);
}

test "Convert from array" {
    var a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var am: [*]f32 = &a;
    var fa: [*]Force(f32) = @ptrCast([*]Force(f32), am);
    try expect(fa[1].y == 5);
}

pub fn QVec(comptime T: type, comptime qVecSize: comptime_int) type {
    return extern struct {
        const Self = @This();
        q: [qVecSize]T,
        pub fn velocity(self: Self, f: Force(T)) Velocity(T) {
            return Velocity(T){
                .x = (1.0 / self.q[header.M01]) * (self.q[header.M02] + 0.5 * f.x),
                .y = (1.0 / self.q[header.M01]) * (self.q[header.M03] + 0.5 * f.y),
                .z = (1.0 / self.q[header.M01]) * (self.q[header.M04] + 0.5 * f.z),
            };
        }
        pub fn zero() Self {
            return Self{
                .q = [_]T{0} ** qVecSize,
            };
        }
    };
}

test "qvec" {
    var z = QVec(f32, 18).zero();
    try expect(z.q[0] == 0);
}

test "velocity" {
    var f = Force(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };
    var qv = QVec(f32, 18){ .q = [_]f32{ 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    var v = qv.velocity(f);
    try expect(v.x == 2.5);
    try expect(v.y == 4);
    try expect(v.z == 5.5);
}

pub fn QVecAccess(comptime T: type, comptime tNi: type) type {
    return extern struct {
        qF: [*]T,
        offset: usize,
        step: usize,
        pub fn q(Self: *const QVecAccess(T, tNi), l: tNi) *T {
            return &Self.qF[Self.offset + Self.step * @intCast(usize, l)];
        }
    };
}

pub fn Streaming(comptime T: type, comptime qVecSize: comptime_int, comptime tNi: type, comptime memoryLayout: header.enum_MemoryLayoutType, comptime streaming: header.enum_Streaming) type {
    return extern struct {
        const Self = @This();
        const FA = FieldAccess(T, qVecSize, tNi, memoryLayout, streaming);
        const h = header;
        //                  h.Q01, h.Q02, h.Q03, h.Q04, h.Q05, h.Q06, h.Q07, h.Q08, h.Q09, h.Q10, h.Q11, h.Q12, h.Q13, h.Q14, h.Q15, h.Q16, h.Q17, h.Q18
        const eso = [_]u32{ h.Q02, h.Q01, h.Q04, h.Q03, h.Q06, h.Q05, h.Q08, h.Q07, h.Q10, h.Q09, h.Q12, h.Q11, h.Q14, h.Q13, h.Q16, h.Q15, h.Q18, h.Q17 };
        evenStep: bool = true,
        inline fn inc(v: tNi, b: tNi) tNi {
            if (v == b) {
                return 1;
            } else {
                return v + 1;
            }
        }
        inline fn esoMap(self: Self, l: u32) u32 {
            if (self.evenStep) {
                return l;
            } else {
                return eso[l];
            }
        }
        pub fn read(self: Self, fa: FA, i: tNi, j: tNi, k: tNi) [qVecSize]T {
            switch (streaming) {
                header.Simple => {
                    var qVec: [qVecSize]T = undefined;
                    var qVecAcc = fa.peek(i, j, k);
                    for (qVec) |_, l| {
                        qVec[l] = qVecAcc.q(@intCast(u32, l)).*;
                    }
                    return qVec;
                },
                header.Esotwist => {
                    var i_ = inc(i, fa.xg - 2);
                    var j_ = inc(j, fa.yg - 2);
                    var k_ = inc(k, fa.zg - 2);
                    var qVec: [qVecSize]T = undefined;
                    var qVecAcc = fa.peek(i, j, k);
                    inline for ([_]u32{ h.Q01, h.Q03, h.Q05, h.Q07, h.Q09, h.Q11 }) |l| {
                        qVec[l] = qVecAcc.q(self.esoMap(l)).*;
                    }
                    qVecAcc = fa.peek(i_, j, k);
                    inline for ([_]u32{ h.Q02, h.Q14, h.Q16 }) |l| {
                        qVec[l] = qVecAcc.q(self.esoMap(l)).*;
                    }
                    qVecAcc = fa.peek(i, j_, k);
                    inline for ([_]u32{ h.Q04, h.Q13, h.Q18 }) |l| {
                        qVec[l] = qVecAcc.q(self.esoMap(l)).*;
                    }
                    qVecAcc = fa.peek(i, j, k_);
                    inline for ([_]u32{ h.Q06, h.Q15, h.Q17 }) |l| {
                        qVec[l] = qVecAcc.q(self.esoMap(l)).*;
                    }
                    qVec[h.Q12] = fa.peek(i, j_, k_).q(self.esoMap(h.Q12)).*;
                    qVec[h.Q10] = fa.peek(i_, j, k_).q(self.esoMap(h.Q10)).*;
                    qVec[h.Q08] = fa.peek(i_, j_, k_).q(self.esoMap(h.Q08)).*;
                    return qVec;
                },
                else => {
                    @compileError("Not supported streaming: '" ++ streaming ++ "'");
                },
            }
        }
        pub fn write(self: Self, fa: FA, i: tNi, j: tNi, k: tNi, qVec: [qVecSize]T) void {
            switch (streaming) {
                header.Simple => {
                    var qVecAcc = fa.peek(i, j, k);
                    for (qVec) |v, l| {
                        qVecAcc.q(@intCast(u32, l)).* = v;
                    }
                },
                header.Esotwist => {
                    var i_ = inc(i, fa.xg - 2);
                    var j_ = inc(j, fa.yg - 2);
                    var k_ = inc(k, fa.zg - 2);
                    var qVecAcc = fa.peek(i, j, k);
                    inline for ([_]u32{ h.Q01, h.Q03, h.Q05, h.Q07, h.Q09, h.Q11 }) |l| {
                        qVecAcc.q(self.esoMap(l)).* = qVec[Self.eso[l]];
                    }
                    qVecAcc = fa.peek(i_, j, k);
                    inline for ([_]u32{ h.Q02, h.Q14, h.Q16 }) |l| {
                        qVecAcc.q(self.esoMap(l)).* = qVec[Self.eso[l]];
                    }
                    qVecAcc = fa.peek(i, j_, k);
                    inline for ([_]u32{ h.Q04, h.Q13, h.Q18 }) |l| {
                        qVecAcc.q(self.esoMap(l)).* = qVec[Self.eso[l]];
                    }
                    qVecAcc = fa.peek(i, j, k_);
                    inline for ([_]u32{ h.Q06, h.Q15, h.Q17 }) |l| {
                        qVecAcc.q(self.esoMap(l)).* = qVec[Self.eso[l]];
                    }
                    fa.peek(i, j_, k_).q(self.esoMap(h.Q12)).* = qVec[Self.eso[h.Q12]];
                    fa.peek(i_, j, k_).q(self.esoMap(h.Q10)).* = qVec[Self.eso[h.Q10]];
                    fa.peek(i_, j_, k).q(self.esoMap(h.Q08)).* = qVec[Self.eso[h.Q08]];
                },
                else => {
                    @compileError("Not supported streaming: '" ++ streaming ++ "'");
                },
            }
        }
        pub fn writeMoments(self: Self, fa: FA, i: tNi, j: tNi, k: tNi, qVec: [qVecSize]T) void {
            switch (streaming) {
                header.Simple => {
                    self.write(fa, i, j, k, qVec);
                },
                header.Esotwist => {
                    var i_ = inc(i, fa.xg - 2);
                    var j_ = inc(j, fa.yg - 2);
                    var k_ = inc(k, fa.zg - 2);
                    var qVecAcc = fa.peek(i, j, k);
                    inline for ([_]u32{ h.Q01, h.Q03, h.Q05, h.Q07, h.Q09, h.Q11 }) |l| {
                        qVecAcc.q(self.esoMap(l)).* = qVec[l];
                    }
                    qVecAcc = fa.peek(i_, j, k);
                    inline for ([_]u32{ h.Q02, h.Q14, h.Q16 }) |l| {
                        qVecAcc.q(self.esoMap(l)).* = qVec[l];
                    }
                    qVecAcc = fa.peek(i, j_, k);
                    inline for ([_]u32{ h.Q04, h.Q13, h.Q18 }) |l| {
                        qVecAcc.q(self.esoMap(l)).* = qVec[l];
                    }
                    qVecAcc = fa.peek(i, j, k_);
                    inline for ([_]u32{ h.Q06, h.Q15, h.Q17 }) |l| {
                        qVecAcc.q(self.esoMap(l)).* = qVec[l];
                    }
                    fa.peek(i, j_, k_).q(self.esoMap(h.Q12)).* = qVec[h.Q12];
                    fa.peek(i_, j, k_).q(self.esoMap(h.Q10)).* = qVec[h.Q10];
                    fa.peek(i_, j_, k).q(self.esoMap(h.Q08)).* = qVec[h.Q08];
                    return qVec;
                },
                else => {
                    @compileError("Not supported streaming: '" ++ streaming ++ "'");
                },
            }
        }
    };
}

pub fn FieldAccess(comptime T: type, comptime qVecSize: comptime_int, comptime tNi: type, comptime memoryLayout: header.enum_MemoryLayoutType, comptime streaming: header.enum_Streaming) type {
    return extern struct {
        const Self = @This();
        const StreamingType = Streaming(T, qVecSize, tNi, memoryLayout, streaming);
        q: [*]T,
        xg: tNi,
        yg: tNi,
        zg: tNi,
        streaming: StreamingType = StreamingType{},

        pub inline fn index(self: Self, i: tNi, j: tNi, k: tNi) usize {
            return @intCast(usize, i * (self.yg * self.zg) + (j * self.zg) + k);
        }

        pub fn peek(self: Self, i: tNi, j: tNi, k: tNi) QVecAccess(T, tNi) {
            switch (memoryLayout) {
                header.MemoryLayoutIJKL => {
                    return QVecAccess(T, tNi){ .qF = self.q, .offset = self.index(i, j, k) * qVecSize, .step = 1 };
                },
                header.MemoryLayoutLIJK => {
                    return QVecAccess(T, tNi){ .qF = self.q, .offset = self.index(i, j, k), .step = self.xg * self.yg * self.zg };
                },
                else => {
                    @compileError("Not supported layout: '" ++ memoryLayout ++ "'");
                },
            }
        }

        pub fn read(self: Self, i: tNi, j: tNi, k: tNi) [qVecSize]T {
            return self.streaming.read(self, i, j, k);
        }

        pub fn write(self: Self, i: tNi, j: tNi, k: tNi, qVec: [qVecSize]T) void {
            return self.streaming.write(self, i, j, k, qVec);
        }

        pub fn writeMoments(self: Self, i: tNi, j: tNi, k: tNi, qVec: [qVecSize]T) void {
            return self.streaming.writeMoments(self, i, j, k, qVec);
        }
    };
}

test "field access" {
    var fQ = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };
    var f = FieldAccess(f32, header.D3Q19, u32, header.MemoryLayoutIJKL, header.Simple){
        .q = &fQ,
        .xg = 2,
        .yg = 1,
        .zg = 1,
    };
    var qa: QVecAccess(f32, u32) = f.peek(1, 0, 0);
    var qv: [header.D3Q19]f32 = f.read(1, 0, 0);
    try expect(qa.q(5).* == 24);
    try expect(qa.q(0).* == 19);
    try expect(qv[5] == 24);
    try expect(qv[0] == 19);
    try expect(f.index(1, 2, 3) == 6);
}

pub fn FlowParams(comptime T: type) type {
    return extern struct {
        const Self = @This();

        initialRho: T = 0,
        reMNonDimensional: T = 0,
        uav: T = 0,

        //ratio mixing length / lattice spacing delta (Smagorinsky)
        cs0: T = 0,

        //compensation of third order terms
        g3: T = 0,

        //kinematic viscosity
        nu: T = 0,

        //forcing in x-direction
        fx0: T = 0,

        //forcing in y-direction
        fy0: T = 0,

        //forcing in z-direction
        fz0: T = 0,

        //Reynolds number based on mean or tip velocity
        Re_m: T = 0,

        //Reynolds number based on the friction velocity uf
        Re_f: T = 0,

        //friction velocity
        uf: T = 0,
        alpha: T = 0,
        beta: T = 0,
        useLES: c_int = 0,

        pub fn create(
            initialRho: T,
            reMNonDimensional: T,
            uav: T,
            cs0: T,
            g3: T,
            nu: T,
            fx0: T,
            fy0: T,
            fz0: T,
            Re_m: T,
            Re_f: T,
            uf: T,
            alpha: T,
            beta: T,
            useLES: c_int,
        ) Self {
            return Self{ .initialRho = initialRho, .reMNonDimensional = reMNonDimensional, .uav = uav, .cs0 = cs0, .g3 = g3, .nu = nu, .fx0 = fx0, .fy0 = fy0, .fz0 = fz0, .Re_m = Re_m, .Re_f = Re_f, .uf = uf, .alpha = alpha, .beta = beta, .useLES = useLES };
        }
    };
}
