pragma circom 2.2.0;
include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/bitify.circom";

// Length/domain-separated, salted vector commitment. Matches worker.cjs exactly.
template VectorCommit(L, DOMAIN) {
    signal input values[L];
    signal input salt;
    signal output root;
    var B = (L + 7) \ 8;
    component init = Poseidon(4);
    init.inputs[0] <== DOMAIN;
    init.inputs[1] <== L;
    init.inputs[2] <== salt;
    init.inputs[3] <== 1;
    signal acc[B+1];
    component folds[B];
    acc[0] <== init.out;
    for (var b=0; b<B; b++) {
        folds[b] = Poseidon(9);
        folds[b].inputs[0] <== acc[b];
        for (var j=0; j<8; j++) {
            if (8*b+j < L) folds[b].inputs[j+1] <== values[8*b+j];
            else folds[b].inputs[j+1] <== 0;
        }
        acc[b+1] <== folds[b].out;
    }
    root <== acc[B];
}

// Only called with already range-constrained unsigned coordinates (5 bits).
template InRegion(LX,HX,LY,HY) {
    signal input x;
    signal input y;
    signal output inside;
    component loX = LessThan(5); component hiX = LessThan(5);
    component loY = LessThan(5); component hiY = LessThan(5);
    loX.in[0] <== x; loX.in[1] <== LX;
    hiX.in[0] <== HX; hiX.in[1] <== x;
    loY.in[0] <== y; loY.in[1] <== LY;
    hiY.in[0] <== HY; hiY.in[1] <== y;
    signal inX; signal inY;
    inX <== (1-loX.out)*(1-hiX.out);
    inY <== (1-loY.out)*(1-hiY.out);
    inside <== inX*inY;
}
