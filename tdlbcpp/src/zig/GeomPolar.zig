pub fn PosPolar(comptime T: type, comptime TQ: type) type {
    //should be exactly the same as
    //struct PosPolar
    return extern struct {

        //    resolution: TQ = 0.0,
        //    rPolar: TQ = 0.0,
        //    tPolar: TQ = 0.0,
        //
        //    iFP: TQ = 0.0,
        //    jFP: TQ = 0.0,
        //    kFP: TQ = 0.0,

        i: T = 0,
        j: T = 0,
        k: T = 0,

        iCartFraction: TQ = 0.0,
        jCartFraction: TQ = 0.0,
        kCartFraction: TQ = 0.0,

        uDelta: TQ = 0.0,
        vDelta: TQ = 0.0,
        wDelta: TQ = 0.0,

        isInternal: bool = false, //Either 0 surface, or 1 solid (the cells between the surface)
    };
}
