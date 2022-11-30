// C interfacing module
const h = @cImport({
    @cDefine("ZIG_IMPORT", "1");
    @cInclude("Header.h");
});
const qVec = @import("QVec.zig");
const compUnit = @import("ComputeUnit.zig");
const collisionES = @import("CollisionEgglesSomers.zig");
const b = @import("Boundary.zig");
const geom = @import("GeomPolar.zig");
const f = @import("Forcing.zig");

// Collision

export fn EgglesSomers_collision_zig__neive__float__long_int__19_ijkl(
    q: [*]f32,
    F: [*]f32,
    Nu: [*]f32,
    xg: c_long,
    yg: c_long,
    zg: c_long,
    flow_cs0: f32,
    flow_g3: f32,
    flow_nu: f32,
    flow_useLes: c_int,
) void {
    var fa = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };

    var flow = qVec.FlowParams(f32){ .nu = flow_nu, .useLES = flow_useLes, .cs0 = flow_cs0, .g3 = flow_g3 };
    var cu = compUnit.ComputeUnit(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple){ .flow = flow, .fa = fa, .F = @ptrCast([*]qVec.Force(f32), F), .Nu = Nu };

    collisionES.EgglesSomers_collision_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple, cu);
}

export fn EgglesSomers_collision_zig__esotwist__float__long_int__19_ijkl(
    q: [*]f32,
    F: [*]f32,
    Nu: [*]f32,
    xg: c_long,
    yg: c_long,
    zg: c_long,
    flow_cs0: f32,
    flow_g3: f32,
    flow_nu: f32,
    flow_useLes: c_int,
    evenStep: c_int,
) void {
    var fa = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };
    fa.streaming.evenStep = if (evenStep == 0) false else true;

    var flow = qVec.FlowParams(f32){ .nu = flow_nu, .useLES = flow_useLes, .cs0 = flow_cs0, .g3 = flow_g3 };
    var cu = compUnit.ComputeUnit(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist){ .flow = flow, .fa = fa, .F = @ptrCast([*]qVec.Force(f32), F), .Nu = Nu };

    collisionES.EgglesSomers_collision_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist, cu);
}

export fn EgglesSomers_collision_moments_zig__neive__float__long_int__19_ijkl(q: [*]f32, xg: c_long, yg: c_long, zg: c_long) void {
    var fa = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };

    collisionES.EgglesSomers_collision_moments_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple, fa);
}

export fn EgglesSomers_collision_moments_zig__esotwist__float__long_int__19_ijkl(q: [*]f32, xg: c_long, yg: c_long, zg: c_long, evenStep: c_int) void {
    var fa = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };
    fa.streaming.evenStep = if (evenStep == 0) false else true;

    collisionES.EgglesSomers_collision_moments_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist, fa);
}

// Boundary

export fn Bounce_zig__neive__float__long_int__19_ijkl(q: [*]f32, xg: c_long, yg: c_long, zg: c_long) void {
    var fa = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };

    b.Bounce_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple, fa);
}

// Forcing
export fn forcing_zig__neive__float__long_int__19_ijkl(
    q: [*]f32,
    F: [*]f32,
    Nu: [*]f32,
    O: [*]bool,
    xg: c_long,
    yg: c_long,
    zg: c_long,
    ghost: c_long,
    flow_cs0: f32,
    flow_g3: f32,
    flow_nu: f32,
    flow_useLes: c_int,
    gs: [*]geom.PosPolar(c_long, f32),
    gsSize: c_int,
    alfa: f32,
    beta: f32,
) void {
    var fa = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };

    var flow = qVec.FlowParams(f32){ .nu = flow_nu, .useLES = flow_useLes, .cs0 = flow_cs0, .g3 = flow_g3 };
    var cu = compUnit.ComputeUnit(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple){ .flow = flow, .fa = fa, .F = @ptrCast([*]qVec.Force(f32), F), .Nu = Nu, .O = O, .ghost = ghost };

    f.forcing_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple, cu, gs, gsSize, alfa, beta);
}

export fn forcing_zig__esotwist__float__long_int__19_ijkl(
    q: [*]f32,
    F: [*]f32,
    Nu: [*]f32,
    O: [*]bool,
    xg: c_long,
    yg: c_long,
    zg: c_long,
    ghost: c_long,
    flow_cs0: f32,
    flow_g3: f32,
    flow_nu: f32,
    flow_useLes: c_int,
    evenStep: c_int,
    gs: [*]geom.PosPolar(c_long, f32),
    gsSize: c_int,
    alfa: f32,
    beta: f32,
) void {
    var fa = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };
    fa.streaming.evenStep = if (evenStep == 0) false else true;

    var flow = qVec.FlowParams(f32){ .nu = flow_nu, .useLES = flow_useLes, .cs0 = flow_cs0, .g3 = flow_g3 };
    var cu = compUnit.ComputeUnit(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist){ .flow = flow, .fa = fa, .F = @ptrCast([*]qVec.Force(f32), F), .Nu = Nu, .O = O, .ghost = ghost};

    f.forcing_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist, cu, gs, gsSize, alfa, beta);
}