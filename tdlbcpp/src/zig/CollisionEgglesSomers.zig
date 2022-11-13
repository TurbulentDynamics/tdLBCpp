const std = @import("std");
const math = std.math;
const h = @cImport({
    @cDefine("ZIG_IMPORT", "1");
    @cInclude("Header.h");
});
const qVec = @import("QVec.zig");

inline fn range(comptime tNi: type, bound: tNi) []const u0 {
    var ubound: usize = if (bound >= 0) @intCast(usize, bound) else 0;
    return [_]u0{0} ** ubound;
}

inline fn EgglesSomers_collision_zig(
    comptime T: type,
    comptime qVecSize: u32,
    comptime tNi: type,
    comptime memoryLayout: h.enum_MemoryLayoutType,
    comptime streaming: h.enum_Streaming,
    flow: qVec.FlowParams(T),
    fq: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming),
    F: [*]qVec.Force(T),
    Nu: [*]T,
) void {
    //kinematic viscosity.
    var b: T = 1.0 / (1.0 + 6 * flow.nu);
    var c: T = 1.0 - 6 * flow.nu;

    var i: tNi = 1;
    while (i < fq.xg - 1) : (i += 1) {
        var j: tNi = 1;
        while (j < fq.yg - 1) : (j += 1) {
            var k: tNi = 1;
            while (k < fq.zg - 1) : (k += 1) {
                var f: *qVec.Force(T) = &F[fq.index(i, j, k)];

                var m: [qVecSize]T = fq.read(i, j, k);
                var mq = qVec.QVec(T, qVecSize){ .q = m };

                var u: qVec.Velocity(T) = mq.velocity(f.*);

                var alpha = [_]T{0} ** qVecSize;

                if (flow.useLES == 1) {
                    var fct: T = 3.0 / (m[h.M01] * (1.0 + 6.0 * (Nu[fq.index(i, j, k)] + flow.nu)));

                    //calculating the derivatives for x, y and z
                    var dudx: T = fct * ((m[h.M02] + 0.5 * F[fq.index(i, j, k)].x * u.x - m[h.M05]));
                    var dvdy: T = fct * ((m[h.M03] + 0.5 * F[fq.index(i, j, k)].y * u.y - m[h.M07]));
                    var dwdz: T = fct * ((m[h.M04] + 0.5 * F[fq.index(i, j, k)].z * u.z - m[h.M10]));

                    var divv: T = dudx + dvdy + dwdz;

                    //calculating the cross terms, used for the shear matrix
                    var dudypdvdx: T = 2 * fct * ((m[h.M03]) + 0.5 * F[fq.index(i, j, k)].y * u.x - m[h.M06]);
                    var dudzpdwdx: T = 2 * fct * ((m[h.M04]) + 0.5 * F[fq.index(i, j, k)].z * u.x - m[h.M08]);
                    var dvdzpdwdy: T = 2 * fct * ((m[h.M04]) + 0.5 * F[fq.index(i, j, k)].z * u.y - m[h.M09]);

                    //calculating sh (the resolved deformation rate, S^2)
                    var sh: T = 2 * dudx * dudx + 2 * dvdy * dvdy + 2 * dwdz * dwdz + dudypdvdx * dudypdvdx + dudzpdwdx * dudzpdwdx + dvdzpdwdy * dvdzpdwdy - (2.0 / 3.0) * divv * divv;

                    //calculating eddy viscosity:
                    //nu_t = (lambda_mix)^2 * sqrt(S^2)     (Smagorinsky)
                    Nu[fq.index(i, j, k)] = flow.cs0 * flow.cs0 * @sqrt(@fabs(sh));

                    // Viscosity is adjusted only for LES, because LES uses a
                    // subgrid-adjustment model for turbulence that's too small to
                    // be captured in the regular cells. This adjustment is
                    // performed by adding the eddy viscosity to the viscosity.
                    // This model is called the Smagorinsky model, however this
                    // implementation is slightly different, as explained by
                    // Somers (1993) -> low strain rates do not excite the
                    // eddy viscosity.

                    var nut: T = Nu[fq.index(i, j, k)] + flow.nu;
                    b = 1.0 / (1.0 + 6 * nut);
                    c = 1.0 - 6 * nut;
                } //end of LES

                //0th order term
                alpha[h.M01] = m[h.M01];

                //1st order term
                alpha[h.M02] = m[h.M02] + f.x;
                alpha[h.M03] = m[h.M03] + f.y;
                alpha[h.M04] = m[h.M04] + f.z;

                //2nd order terms
                alpha[h.M05] = (2.0 * (m[h.M02] + 0.5 * f.x) * u.x - m[h.M05] * c) * b;
                alpha[h.M06] = (2.0 * (m[h.M02] + 0.5 * f.x) * u.y - m[h.M06] * c) * b;
                alpha[h.M07] = (2.0 * (m[h.M03] + 0.5 * f.y) * u.y - m[h.M07] * c) * b;

                alpha[h.M08] = (2.0 * (m[h.M02] + 0.5 * f.x) * u.z - m[h.M08] * c) * b;
                alpha[h.M09] = (2.0 * (m[h.M03] + 0.5 * f.y) * u.z - m[h.M09] * c) * b;
                alpha[h.M10] = (2.0 * (m[h.M04] + 0.5 * f.z) * u.z - m[h.M10] * c) * b;

                //3rd order terms
                alpha[h.M11] = -flow.g3 * m[h.M11];
                alpha[h.M12] = -flow.g3 * m[h.M12];
                alpha[h.M13] = -flow.g3 * m[h.M13];
                alpha[h.M14] = -flow.g3 * m[h.M14];
                alpha[h.M15] = -flow.g3 * m[h.M15];
                alpha[h.M16] = -flow.g3 * m[h.M16];

                //4th order terms
                alpha[h.M17] = 0.0;
                alpha[h.M18] = 0.0;

                // Start of invMoments, which is responsible for determining
                // the N-field (x) from alpha+ (alpha). It does this by using eq.
                // 12 in the article by Eggels and Somers (1995), which means
                // it's using the "filter matrix E" (not really present in the
                // code as matrix, but it's where the coefficients come from).

                var l: usize = 1;
                while (l < qVecSize) : (l += 1) {
                    alpha[l] /= 24.0;
                }

                m[h.Q01] = 2 * alpha[h.M01] + 4 * alpha[h.M02] + 3 * alpha[h.M05] - 3 * alpha[h.M07] - 3 * alpha[h.M10] - 2 * alpha[h.M11] - 2 * alpha[h.M13] + 2 * alpha[h.M17] + 2 * alpha[h.M18];

                m[h.Q02] = 2 * alpha[h.M01] - 4 * alpha[h.M02] + 3 * alpha[h.M05] - 3 * alpha[h.M07] - 3 * alpha[h.M10] + 2 * alpha[h.M11] + 2 * alpha[h.M13] + 2 * alpha[h.M17] + 2 * alpha[h.M18];

                m[h.Q03] = 2 * alpha[h.M01] + 4 * alpha[h.M03] - 3 * alpha[h.M05] + 3 * alpha[h.M07] - 3 * alpha[h.M10] - 2 * alpha[h.M12] - 2 * alpha[h.M14] + 2 * alpha[h.M17] - 2 * alpha[h.M18];

                m[h.Q04] = 2 * alpha[h.M01] - 4 * alpha[h.M03] - 3 * alpha[h.M05] + 3 * alpha[h.M07] - 3 * alpha[h.M10] + 2 * alpha[h.M12] + 2 * alpha[h.M14] + 2 * alpha[h.M17] - 2 * alpha[h.M18];

                m[h.Q05] = 2 * alpha[h.M01] + 4 * alpha[h.M04] - 3 * alpha[h.M05] - 3 * alpha[h.M07] + 3 * alpha[h.M10] - 4 * alpha[h.M15] - 4 * alpha[h.M17];

                m[h.Q06] = 2 * alpha[h.M01] - 4 * alpha[h.M04] - 3 * alpha[h.M05] - 3 * alpha[h.M07] + 3 * alpha[h.M10] + 4 * alpha[h.M15] - 4 * alpha[h.M17];

                m[h.Q07] = alpha[h.M01] + 2 * alpha[h.M02] + 2 * alpha[h.M03] + 1.5 * alpha[h.M05] + 6 * alpha[h.M06] + 1.5 * alpha[h.M07] - 1.5 * alpha[h.M10] + 2 * alpha[h.M11] + 2 * alpha[h.M12] - 2 * alpha[h.M17];

                m[h.M14] = alpha[h.M01] - 2 * alpha[h.M02] + 2 * alpha[h.M03] + 1.5 * alpha[h.M05] - 6 * alpha[h.M06] + 1.5 * alpha[h.M07] - 1.5 * alpha[h.M10] - 2 * alpha[h.M11] + 2 * alpha[h.M12] - 2 * alpha[h.M17];

                m[h.M08] = alpha[h.M01] - 2 * alpha[h.M02] - 2 * alpha[h.M03] + 1.5 * alpha[h.M05] + 6 * alpha[h.M06] + 1.5 * alpha[h.M07] - 1.5 * alpha[h.M10] - 2 * alpha[h.M11] - 2 * alpha[h.M12] - 2 * alpha[h.M17];

                m[h.M13] = alpha[h.M01] + 2 * alpha[h.M02] - 2 * alpha[h.M03] + 1.5 * alpha[h.M05] - 6 * alpha[h.M06] + 1.5 * alpha[h.M07] - 1.5 * alpha[h.M10] + 2 * alpha[h.M11] - 2 * alpha[h.M12] - 2 * alpha[h.M17];

                m[h.M09] = alpha[h.M01] + 2 * alpha[h.M02] + 2 * alpha[h.M04] + 1.5 * alpha[h.M05] - 1.5 * alpha[h.M07] + 6 * alpha[h.M08] + 1.5 * alpha[h.M10] - alpha[h.M11] + alpha[h.M13] + alpha[h.M15] - alpha[h.M16] + alpha[h.M17] - alpha[h.M18];

                m[h.M16] = alpha[h.M01] - 2 * alpha[h.M02] + 2 * alpha[h.M04] + 1.5 * alpha[h.M05] - 1.5 * alpha[h.M07] - 6 * alpha[h.M08] + 1.5 * alpha[h.M10] + alpha[h.M11] - alpha[h.M13] + alpha[h.M15] - alpha[h.M16] + alpha[h.M17] - alpha[h.M18];

                m[h.M10] = alpha[h.M01] - 2 * alpha[h.M02] - 2 * alpha[h.M04] + 1.5 * alpha[h.M05] - 1.5 * alpha[h.M07] + 6 * alpha[h.M08] + 1.5 * alpha[h.M10] + alpha[h.M11] - alpha[h.M13] - alpha[h.M15] + alpha[h.M16] + alpha[h.M17] - alpha[h.M18];

                m[h.M15] = alpha[h.M01] + 2 * alpha[h.M02] - 2 * alpha[h.M04] + 1.5 * alpha[h.M05] - 1.5 * alpha[h.M07] - 6 * alpha[h.M08] + 1.5 * alpha[h.M10] - alpha[h.M11] + alpha[h.M13] - alpha[h.M15] + alpha[h.M16] + alpha[h.M17] - alpha[h.M18];

                m[h.M11] = alpha[h.M01] + 2 * alpha[h.M03] + 2 * alpha[h.M04] - 1.5 * alpha[h.M05] + 1.5 * alpha[h.M07] + 6 * alpha[h.M09] + 1.5 * alpha[h.M10] - alpha[h.M12] + alpha[h.M14] + alpha[h.M15] + alpha[h.M16] + alpha[h.M17] + alpha[h.M18];

                m[h.M18] = alpha[h.M01] - 2 * alpha[h.M03] + 2 * alpha[h.M04] - 1.5 * alpha[h.M05] + 1.5 * alpha[h.M07] - 6 * alpha[h.M09] + 1.5 * alpha[h.M10] + alpha[h.M12] - alpha[h.M14] + alpha[h.M15] + alpha[h.M16] + alpha[h.M17] + alpha[h.M18];

                m[h.M12] = alpha[h.M01] - 2 * alpha[h.M03] - 2 * alpha[h.M04] - 1.5 * alpha[h.M05] + 1.5 * alpha[h.M07] + 6 * alpha[h.M09] + 1.5 * alpha[h.M10] + alpha[h.M12] - alpha[h.M14] - alpha[h.M15] - alpha[h.M16] + alpha[h.M17] + alpha[h.M18];

                m[h.M17] = alpha[h.M01] + 2 * alpha[h.M03] - 2 * alpha[h.M04] - 1.5 * alpha[h.M05] + 1.5 * alpha[h.M07] - 6 * alpha[h.M09] + 1.5 * alpha[h.M10] - alpha[h.M12] + alpha[h.M14] - alpha[h.M15] - alpha[h.M16] + alpha[h.M17] + alpha[h.M18];

                fq.write(i, j, k, m);
            }
        }
    }
}

inline fn EgglesSomers_collision_moments_zig(
    comptime T: type,
    comptime qVecSize: u32,
    comptime tNi: type,
    comptime memoryLayout: h.enum_MemoryLayoutType,
    comptime streaming: h.enum_Streaming,
    fq: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming),
) void {

    var i: tNi = 1;
    while (i < fq.xg - 1) : (i += 1) {
        var j: tNi = 1;
        while (j < fq.yg - 1) : (j += 1) {
            var k: tNi = 1;
            while (k < fq.zg - 1) : (k += 1) {
                var q: [qVecSize]T = fq.read(i, j, k);
                var m = [_]T{0} ** qVecSize;

                //the first position is simply the entire mass-vector (Q summed up)
                m[h.M01] = q[h.Q01] + q[h.Q03] + q[h.Q02] + q[h.Q04] + q[h.Q05] + q[h.Q06] + q[h.Q07] + q[h.Q14] + q[h.Q08] + q[h.Q13] + q[h.Q09] + q[h.Q16] + q[h.Q10] + q[h.Q15] + q[h.Q11] + q[h.Q18] + q[h.Q12] + q[h.Q17];


                //the second position is everything with an x-component
                m[h.M02] = q[h.Q01] - q[h.Q02] + q[h.Q07] - q[h.Q14] - q[h.Q08] + q[h.Q13] + q[h.Q09] - q[h.Q16] - q[h.Q10] + q[h.Q15];


                //the third position is everything with an y-component
                m[h.M03] = q[h.Q03] - q[h.Q04] + q[h.Q07] + q[h.Q14] - q[h.Q08] - q[h.Q13] + q[h.Q11] - q[h.Q18] - q[h.Q12] + q[h.Q17];


                //the fourth position is everything with a z-component
                m[h.M04] = q[h.Q05] - q[h.Q06] + q[h.Q09] + q[h.Q16] - q[h.Q10] - q[h.Q15] + q[h.Q11] + q[h.Q18] - q[h.Q12] - q[h.Q17];


                //starting from the fifth position, it gets more complicated in
                //structure, but it still follows the article by Eggels and Somers
                m[h.M05] =  - q[h.Q03] - q[h.Q04] - q[h.Q05] - q[h.Q06] + q[h.Q07] + q[h.Q14] + q[h.Q08] + q[h.Q13] + q[h.Q09] + q[h.Q16] + q[h.Q10] + q[h.Q15];


                m[h.M06] = q[h.Q07] - q[h.Q14] + q[h.Q08] - q[h.Q13];

                m[h.M07] =  - q[h.Q01] - q[h.Q02] - q[h.Q05] - q[h.Q06] + q[h.Q07] + q[h.Q14] + q[h.Q08] + q[h.Q13] + q[h.Q11] + q[h.Q18] + q[h.Q12] + q[h.Q17];

                m[h.M08] = q[h.Q09] - q[h.Q16] + q[h.Q10] - q[h.Q15];

                m[h.M09] = q[h.Q11] - q[h.Q18] + q[h.Q12] - q[h.Q17];

                m[h.M10] =  - q[h.Q01] - q[h.Q03] - q[h.Q02] - q[h.Q04] + q[h.Q09] + q[h.Q16] + q[h.Q10] + q[h.Q15] + q[h.Q11] + q[h.Q18] + q[h.Q12] + q[h.Q17];

                m[h.M11] =  - q[h.Q01] + q[h.Q02] + 2*q[h.Q07] - 2*q[h.Q14] - 2*q[h.Q08] + 2*q[h.Q13] - q[h.Q09] + q[h.Q16] + q[h.Q10] - q[h.Q15];

                m[h.M12] =  - q[h.Q03] + q[h.Q04] + 2*q[h.Q07] + 2*q[h.Q14] - 2*q[h.Q08] - 2*q[h.Q13] - q[h.Q11] + q[h.Q18] + q[h.Q12] - q[h.Q17];

                m[h.M13] =  - 3*q[h.Q01] + 3*q[h.Q02] + 3*q[h.Q09] - 3*q[h.Q16] - 3*q[h.Q10] + 3*q[h.Q15];

                m[h.M14] =  - 3*q[h.Q03] + 3*q[h.Q04] + 3*q[h.Q11] - 3*q[h.Q18] - 3*q[h.Q12] + 3*q[h.Q17];

                m[h.M15] =  - 2*q[h.Q05] + 2*q[h.Q06] + q[h.Q09] + q[h.Q16] - q[h.Q10] - q[h.Q15] + q[h.Q11] + q[h.Q18] - q[h.Q12] - q[h.Q17];

                m[h.M16] =  - 3*q[h.Q09] - 3*q[h.Q16] + 3*q[h.Q10] + 3*q[h.Q15] + 3*q[h.Q11] + 3*q[h.Q18] - 3*q[h.Q12] - 3*q[h.Q17];

                m[h.M17] = 0.5*q[h.Q01] + 0.5*q[h.Q03] + 0.5*q[h.Q02] + 0.5*q[h.Q04] - q[h.Q05] - q[h.Q06] - q[h.Q07] - q[h.Q14] - q[h.Q08] - q[h.Q13] + 0.5*q[h.Q09] + 0.5*q[h.Q16] + 0.5*q[h.Q10] + 0.5*q[h.Q15] + 0.5*q[h.Q11] + 0.5*q[h.Q18] + 0.5*q[h.Q12] + 0.5*q[h.Q17];

                m[h.M18] = 1.5*q[h.Q01] - 1.5*q[h.Q03] + 1.5*q[h.Q02] - 1.5*q[h.Q04] - 1.5*q[h.Q09] - 1.5*q[h.Q16] - 1.5*q[h.Q10] - 1.5*q[h.Q15] + 1.5*q[h.Q11] + 1.5*q[h.Q18] + 1.5*q[h.Q12] + 1.5*q[h.Q17];

                fq.writeMoments(i, j, k, m);

            }
        }
    }

}//end of func



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
    var fq = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };

    var flow = qVec.FlowParams(f32){ .nu = flow_nu, .useLES = flow_useLes, .cs0 = flow_cs0, .g3 = flow_g3 };

    EgglesSomers_collision_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple, flow, fq, @ptrCast([*]qVec.Force(f32), F), Nu);
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
    var fq = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };
    fq.streaming.evenStep = if (evenStep == 0) false else true;

    var flow = qVec.FlowParams(f32){ .nu = flow_nu, .useLES = flow_useLes, .cs0 = flow_cs0, .g3 = flow_g3 };

    EgglesSomers_collision_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist, flow, fq, @ptrCast([*]qVec.Force(f32), F), Nu);
}

export fn EgglesSomers_collision_moments_zig__neive__float__long_int__19_ijkl(
    q: [*]f32,
    xg: c_long,
    yg: c_long,
    zg: c_long
) void {
    var fq = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };

    EgglesSomers_collision_moments_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Simple, fq);
}

export fn EgglesSomers_collision_moments_zig__esotwist__float__long_int__19_ijkl(
    q: [*]f32,
    xg: c_long,
    yg: c_long,
    zg: c_long,
    evenStep: c_int
) void {
    var fq = qVec.FieldAccess(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist){
        .q = q,
        .xg = xg,
        .yg = yg,
        .zg = zg,
    };
    fq.streaming.evenStep = if (evenStep == 0) false else true;

    EgglesSomers_collision_moments_zig(f32, 18, c_long, h.MemoryLayoutIJKL, h.Esotwist, fq);
}
