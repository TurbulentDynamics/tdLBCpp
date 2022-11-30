//
//  Forcing.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

const std = @import("std");
const geom = @import("GeomPolar.zig");
const h = @cImport({
    @cDefine("ZIG_IMPORT", "1");
    @cInclude("Header.h");
});
const qVec = @import("QVec.zig");
const cu = @import("ComputeUnit.zig");

inline fn calcWeight(comptime T: type, xP: T) T {
    var weight: T = 0;

    var x: T = @fabs(xP); //NB abs returns an interger, fabs returns a float

    if (x <= 1.5) {
        if (x <= 0.5) {
            weight = (1.0 / 3.0) * (1.0 + @sqrt(-3.0 * x * x + 1.0));
        } else {
            weight = (1.0 / 6.0) * (5.0 - (3.0 * x) - @sqrt((-3.0 * ((1.0 - x) * (1.0 - x))) + 1.0));
        }
    }

    return weight;
}

inline fn smoothedDeltaFunction(comptime T: type, comptime tNi: type, i_cart_fraction: T, k_cart_fraction: T, ppp: *[3][3]T) void {
    var k: tNi = -1;
    while (k <= 1) : (k += 1) {
        var i: tNi = -1;
        while (i <= 1) : (i += 1) {
            var hx: T = -i_cart_fraction + @intToFloat(T, i);
            var hz: T = -k_cart_fraction + @intToFloat(T, k);

            ppp.*[@intCast(usize, i + 1)][@intCast(usize, k + 1)] = calcWeight(T, hx) * calcWeight(T, hz);
        }
    }
}

pub inline fn forcing_zig(comptime T: type, comptime qVecSize: u32, comptime tNi: type, comptime memoryLayout: h.enum_MemoryLayoutType, comptime streaming: h.enum_Streaming, compUnit: cu.ComputeUnit(T, qVecSize, tNi, memoryLayout, streaming), gs: [*]geom.PosPolar(tNi, T), gsSize: c_int, alfa: T, beta: T) void {
    const fa = compUnit.fa;
    const O = compUnit.O;
    const F = compUnit.F;

    @memset(@ptrCast([*]u8, O), 0, @intCast(usize, @sizeOf(bool) * fa.xg * fa.yg * fa.zg));

    var gsi: usize = 0;
    while (gsi < gsSize) : (gsi += 1) {
        var g: geom.PosPolar(tNi, T) = gs[gsi];

        var ppp: [3][3]T = [_][3]T{[_]T{0} ** 3} ** 3;

        var i: tNi = g.i + compUnit.ghost;
        var j: tNi = g.j + compUnit.ghost;
        var k: tNi = g.k + compUnit.ghost;
        if ((i < 1) or (i > fa.xg - 1) or (j < 1) or (j > fa.yg - 1) or (k < 1) or (k > fa.zg - 1)) {
            continue;
        }

        smoothedDeltaFunction(T, tNi, g.iCartFraction, g.kCartFraction, &ppp);

        var rhoSum: T = 0.0;
        var xSum: T = 0.0;
        var ySum: T = 0.0;
        var zSum: T = 0.0;

        var k1: tNi = -1;
        while (k1 <= 1) : (k1 += 1) {
            var k1i = @intCast(usize, k1 + 1);
            var _i1: tNi = -1;
            while (_i1 <= 1) : (_i1 += 1) {
                var i1i = @intCast(usize, _i1 + 1);
                var _i2: tNi = i + _i1;
                var j2: tNi = j;
                var k2: tNi = k + k1;

                if (_i2 == 0) _i2 = fa.xg - 2;
                if (_i2 == fa.xg - 1) _i2 = 1;
                if (k2 == 0) k2 = fa.zg - 2;
                if (k2 == fa.zg - 1) k2 = 1;

                var q: [qVecSize]T = fa.read(_i2, j2, k2);
                var rho: T = q[h.M01];

                var localForce: *qVec.Force(T) = &F[fa.index(_i2, j2, k2)];

                var x: T = q[h.M02] + 0.5 * localForce.*.x;
                var y: T = q[h.M03] + 0.5 * localForce.*.y;
                var z: T = q[h.M04] + 0.5 * localForce.*.z;

                //adding the density of a nearby point using a weight (in ppp)
                rhoSum += ppp[i1i][k1i] * rho;

                //adding the velocity of a nearby point using a weight (in ppp)
                xSum += ppp[i1i][k1i] * x;
                ySum += ppp[i1i][k1i] * y;
                zSum += ppp[i1i][k1i] * z;
            }
        } //endfor  j1, k1

        //calculating the difference between the actual (weighted) speed and
        //the required (no-slip) velocity
        xSum -= rhoSum * g.uDelta;
        ySum -= rhoSum * g.vDelta;
        zSum -= rhoSum * g.wDelta;

        k1 = -1;
        while (k1 <= 1) : (k1 += 1) {
            var k1i = @intCast(usize, k1 + 1);
            var _i1: tNi = -1;
            while (_i1 <= 1) : (_i1 += 1) {
                var i1i = @intCast(usize, _i1 + 1);
                var _i2: tNi = i + _i1;
                var j2: tNi = j;
                var k2: tNi = k + k1;

                if (_i2 == 0) _i2 = fa.xg - 2;
                if (_i2 == fa.xg - 1) _i2 = 1;
                if (k2 == 0) k2 = fa.zg - 2;
                if (k2 == fa.zg - 1) k2 = 1;

                var localForce: qVec.Force(T) = F[fa.index(_i2, j2, k2)];

                (&F[fa.index(_i2, j2, k2)]).*.x = alfa * localForce.x - beta * ppp[i1i][k1i] * xSum;
                (&F[fa.index(_i2, j2, k2)]).*.y = alfa * localForce.y - beta * ppp[i1i][k1i] * ySum;
                (&F[fa.index(_i2, j2, k2)]).*.z = alfa * localForce.z - beta * ppp[i1i][k1i] * zSum;

                O[fa.index(_i2, j2, k2)] = true;
            }
        } //endfor  j1, k1

    } //endfor

    var i: tNi = 1;
    while (i < fa.xg) : (i += 1) {
        var j: tNi = 1;
        while (j < fa.yg) : (j += 1) {
            var k: tNi = 1;
            while (k < fa.zg) : (k += 1) {
                if (!O[fa.index(i, j, k)]) {
                    (&F[fa.index(i, j, k)]).*.x = 0.0;
                    (&F[fa.index(i, j, k)]).*.y = 0.0;
                    (&F[fa.index(i, j, k)]).*.z = 0.0;
                } else {
                    //Set it back to 0
                    O[fa.index(i, j, k)] = false;
                } //endif
            }
        }
    } //endfor  ijk

} //end of func

