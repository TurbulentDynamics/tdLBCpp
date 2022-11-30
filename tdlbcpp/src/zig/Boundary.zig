const h = @cImport({
    @cDefine("ZIG_IMPORT", "1");
    @cInclude("Header.h");
});
const qVec = @import("QVec.zig");

pub inline fn Bounce_zig(comptime T: type, comptime qVecSize: u32, comptime tNi: type, comptime memoryLayout: h.enum_MemoryLayoutType, comptime streaming: h.enum_Streaming, fa: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming)) void {

    //Takes the vector from the active cell, reverses it, and places it in the
    //ghost cell (the streaming function can then operate on the ghost cell to
    //bring it back to the active cell

    bounceBackBoundaryRight(T, qVecSize, tNi, memoryLayout, streaming, fa);
    bounceBackBoundaryLeft(T, qVecSize, tNi, memoryLayout, streaming, fa);
    bounceBackBoundaryUp(T, qVecSize, tNi, memoryLayout, streaming, fa);
    bounceBackBoundaryDown(T, qVecSize, tNi, memoryLayout, streaming, fa);
    bounceBackBoundaryBackward(T, qVecSize, tNi, memoryLayout, streaming, fa);
    bounceBackBoundaryForward(T, qVecSize, tNi, memoryLayout, streaming, fa);

    //Needs to be separated into each edge.
    bounceBackEdges(T, qVecSize, tNi, memoryLayout, streaming, fa);
}

inline fn bounceBackEdges(comptime T: type, comptime qVecSize: u32, comptime tNi: type, comptime memoryLayout: h.enum_MemoryLayoutType, comptime streaming: h.enum_Streaming, fa: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming)) void {
    var i: tNi = 0;
    var j: tNi = 0;
    var k: tNi = 0;

    i = 0;
    j = 0;
    k = 1;
    while (k < fa.zg - 1) : (k += 1) {
        fa.peek(i, j, k).q(h.Q07).* = fa.peekDirn(h.Q07, i, j, k).q(h.Q08).*;
    }

    i = fa.xg - 1;
    j = 0;
    k = 1;
    while (k < fa.zg - 1) : (k += 1) {
        fa.peek(i, j, k).q(h.Q14).* = fa.peekDirn(h.Q14, i, j, k).q(h.Q13).*;
    }

    i = 0;
    j = fa.yg - 1;
    k = 1;
    while (k < fa.zg - 1) : (k += 1) {
        fa.peek(i, j, k).q(h.Q13).* = fa.peekDirn(h.Q13, i, j, k).q(h.Q14).*;
    }

    i = fa.xg - 1;
    j = fa.yg - 1;
    k = 1;
    while (k < fa.zg - 1) : (k += 1) {
        fa.peek(i, j, k).q(h.Q08).* = fa.peekDirn(h.Q08, i, j, k).q(h.Q07).*;
    }

    i = 0;
    k = 0;
    j = 1;
    while (j < fa.yg - 1) : (j += 1) {
        fa.peek(i, j, k).q(h.Q09).* = fa.peekDirn(h.Q09, i, j, k).q(h.Q10).*;
    }

    i = 0;
    k = fa.zg - 1;
    j = 1;
    while (j < fa.yg - 1) : (j += 1) {
        fa.peek(i, j, k).q(h.Q15).* = fa.peekDirn(h.Q15, i, j, k).q(h.Q16).*;
    }

    i = fa.xg - 1;
    k = fa.zg - 1;
    j = 1;
    while (j < fa.yg - 1) : (j += 1) {
        fa.peek(i, j, k).q(h.Q10).* = fa.peekDirn(h.Q10, i, j, k).q(h.Q09).*;
    }

    i = fa.xg - 1;
    k = 0;
    j = 1;
    while (j < fa.yg - 1) : (j += 1) {
        fa.peek(i, j, k).q(h.Q16).* = fa.peekDirn(h.Q16, i, j, k).q(h.Q15).*;
    }

    j = 0;
    k = 0;
    i = 1;
    while (i < fa.xg - 1) : (i += 1) {
        fa.peek(i, j, k).q(h.Q11).* = fa.peekDirn(h.Q11, i, j, k).q(h.Q12).*;
    }

    j = 0;
    k = fa.zg - 1;
    i = 1;
    while (i < fa.xg - 1) : (i += 1) {
        fa.peek(i, j, k).q(h.Q17).* = fa.peekDirn(h.Q17, i, j, k).q(h.Q18).*;
    }

    j = fa.yg - 1;
    k = fa.zg - 1;
    i = 1;
    while (i < fa.xg - 1) : (i += 1) {
        fa.peek(i, j, k).q(h.Q12).* = fa.peekDirn(h.Q12, i, j, k).q(h.Q11).*;
    }

    j = fa.yg - 1;
    k = 0;
    i = 1;
    while (i < fa.xg - 1) : (i += 1) {
        fa.peek(i, j, k).q(h.Q18).* = fa.peekDirn(h.Q18, i, j, k).q(h.Q17).*;
    }
}

inline fn bounceBackBoundaryRight(comptime T: type, comptime qVecSize: u32, comptime tNi: type, comptime memoryLayout: h.enum_MemoryLayoutType, comptime streaming: h.enum_Streaming, fa: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming)) void {

    //dest = source

    var j: tNi = 1;
    while (j < fa.yg - 1) : (j += 1) {
        var k: tNi = 1;
        while (k < fa.zg - 1) : (k += 1) {
            var i: tNi = 0;

            fa.peek(i, j, k).q(h.Q01).* = fa.peekDirn(h.Q01, i, j, k).q(h.Q02).*;
            fa.peek(i, j, k).q(h.Q07).* = fa.peekDirn(h.Q07, i, j, k).q(h.Q08).*;
            fa.peek(i, j, k).q(h.Q13).* = fa.peekDirn(h.Q13, i, j, k).q(h.Q14).*;
            fa.peek(i, j, k).q(h.Q09).* = fa.peekDirn(h.Q09, i, j, k).q(h.Q10).*;
            fa.peek(i, j, k).q(h.Q15).* = fa.peekDirn(h.Q15, i, j, k).q(h.Q16).*;
        }
    }
}

inline fn bounceBackBoundaryLeft(comptime T: type, comptime qVecSize: u32, comptime tNi: type, comptime memoryLayout: h.enum_MemoryLayoutType, comptime streaming: h.enum_Streaming, fa: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming)) void {
    var j: tNi = 1;
    while (j < fa.yg - 1) : (j += 1) {
        var k: tNi = 1;
        while (k < fa.zg - 1) : (k += 1) {
            var i: tNi = fa.xg - 1;

            fa.peek(i, j, k).q(h.Q02).* = fa.peekDirn(h.Q02, i, j, k).q(h.Q01).*;
            fa.peek(i, j, k).q(h.Q08).* = fa.peekDirn(h.Q08, i, j, k).q(h.Q07).*;
            fa.peek(i, j, k).q(h.Q14).* = fa.peekDirn(h.Q14, i, j, k).q(h.Q13).*;
            fa.peek(i, j, k).q(h.Q10).* = fa.peekDirn(h.Q10, i, j, k).q(h.Q09).*;
            fa.peek(i, j, k).q(h.Q16).* = fa.peekDirn(h.Q16, i, j, k).q(h.Q15).*;
        }
    }
}

inline fn bounceBackBoundaryUp(comptime T: type, comptime qVecSize: u32, comptime tNi: type, comptime memoryLayout: h.enum_MemoryLayoutType, comptime streaming: h.enum_Streaming, fa: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming)) void {

    // TODO:  Check xg1 here
    var i: tNi = 1;
    while (i < fa.xg - 1) : (i += 1) {
        var k: tNi = 1;
        while (k < fa.zg - 1) : (k += 1) {
            var j: tNi = 0;

            fa.peek(i, j, k).q(h.Q03).* = fa.peekDirn(h.Q03, i, j, k).q(h.Q04).*;
            fa.peek(i, j, k).q(h.Q07).* = fa.peekDirn(h.Q07, i, j, k).q(h.Q08).*;
            fa.peek(i, j, k).q(h.Q14).* = fa.peekDirn(h.Q14, i, j, k).q(h.Q13).*;
            fa.peek(i, j, k).q(h.Q11).* = fa.peekDirn(h.Q11, i, j, k).q(h.Q12).*;
            fa.peek(i, j, k).q(h.Q17).* = fa.peekDirn(h.Q17, i, j, k).q(h.Q18).*;
        }
    }
}

inline fn bounceBackBoundaryDown(comptime T: type, comptime qVecSize: u32, comptime tNi: type, comptime memoryLayout: h.enum_MemoryLayoutType, comptime streaming: h.enum_Streaming, fa: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming)) void {
    var i: tNi = 1;
    while (i < fa.xg - 1) : (i += 1) {
        var k: tNi = 1;
        while (k < fa.zg - 1) : (k += 1) {
            var j: tNi = fa.yg - 1;

            fa.peek(i, j, k).q(h.Q04).* = fa.peekDirn(h.Q04, i, j, k).q(h.Q03).*;
            fa.peek(i, j, k).q(h.Q08).* = fa.peekDirn(h.Q08, i, j, k).q(h.Q07).*;
            fa.peek(i, j, k).q(h.Q13).* = fa.peekDirn(h.Q13, i, j, k).q(h.Q14).*;
            fa.peek(i, j, k).q(h.Q12).* = fa.peekDirn(h.Q12, i, j, k).q(h.Q11).*;
            fa.peek(i, j, k).q(h.Q18).* = fa.peekDirn(h.Q18, i, j, k).q(h.Q17).*;
        }
    }
}

inline fn bounceBackBoundaryBackward(comptime T: type, comptime qVecSize: u32, comptime tNi: type, comptime memoryLayout: h.enum_MemoryLayoutType, comptime streaming: h.enum_Streaming, fa: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming)) void {
    var i: tNi = 1;
    while (i < fa.xg - 1) : (i += 1) {
        var j: tNi = 1;
        while (j < fa.yg - 1) : (j += 1) {
            var k: tNi = 0;

            fa.peek(i, j, k).q(h.Q05).* = fa.peekDirn(h.Q05, i, j, k).q(h.Q06).*;
            fa.peek(i, j, k).q(h.Q09).* = fa.peekDirn(h.Q09, i, j, k).q(h.Q10).*;
            fa.peek(i, j, k).q(h.Q16).* = fa.peekDirn(h.Q16, i, j, k).q(h.Q15).*;
            fa.peek(i, j, k).q(h.Q11).* = fa.peekDirn(h.Q11, i, j, k).q(h.Q12).*;
            fa.peek(i, j, k).q(h.Q18).* = fa.peekDirn(h.Q18, i, j, k).q(h.Q17).*;
        }
    }
}

inline fn bounceBackBoundaryForward(comptime T: type, comptime qVecSize: u32, comptime tNi: type, comptime memoryLayout: h.enum_MemoryLayoutType, comptime streaming: h.enum_Streaming, fa: qVec.FieldAccess(T, qVecSize, tNi, memoryLayout, streaming)) void {
    var i: tNi = 1;
    while (i < fa.xg - 1) : (i += 1) {
        var j: tNi = 1;
        while (j < fa.yg - 1) : (j += 1) {
            var k: tNi = fa.zg - 1;

            fa.peek(i, j, k).q(h.Q06).* = fa.peekDirn(h.Q06, i, j, k).q(h.Q05).*;
            fa.peek(i, j, k).q(h.Q10).* = fa.peekDirn(h.Q10, i, j, k).q(h.Q09).*;
            fa.peek(i, j, k).q(h.Q15).* = fa.peekDirn(h.Q15, i, j, k).q(h.Q16).*;
            fa.peek(i, j, k).q(h.Q12).* = fa.peekDirn(h.Q12, i, j, k).q(h.Q11).*;
            fa.peek(i, j, k).q(h.Q17).* = fa.peekDirn(h.Q17, i, j, k).q(h.Q18).*;
        }
    }
}
