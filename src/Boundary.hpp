//
//  ComputeGroup.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"





template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::bounceBackBoundary(){
    
    //DO BOUNCE BACK TOP
    
    //J is +ve downwards
        
    
    for (tNi i = 1; i<=xg1; i++){
        tNi j = 1;
        for (tNi k = 1; k<=zg1; k++){

                
            //Takes the vector from the active cell, reverses it, and places it in the
            //ghost cell (the streaming function can then operate on the ghost cell to
            //bring it back to the active cell
            
            //dest = source
                
            Q[dirnQ4(i,j,k)].q[Q4] = Q[dirnQ000(i,j,k)].q[Q3];
            
            Q[dirnQ8(i,j,k)].q[Q8] = Q[dirnQ000(i,j,k)].q[Q7];
            Q[dirnQ12(i,j,k)].q[Q12] = Q[dirnQ000(i,j,k)].q[Q11];
            Q[dirnQ13(i,j,k)].q[Q13] = Q[dirnQ000(i,j,k)].q[Q14];
            Q[dirnQ16(i,j,k)].q[Q16] = Q[dirnQ000(i,j,k)].q[Q17];

            Q[dirnQ20(i,j,k)].q[Q20] = Q[dirnQ000(i,j,k)].q[Q19];
            Q[dirnQ22(i,j,k)].q[Q22] = Q[dirnQ000(i,j,k)].q[Q21];
            Q[dirnQ23(i,j,k)].q[Q23] = Q[dirnQ000(i,j,k)].q[Q24];
            Q[dirnQ26(i,j,k)].q[Q26] = Q[dirnQ000(i,j,k)].q[Q25];

        }
    }
    
    
    
    //
    //    if (grid.ngx > 1) exit(1);
    //
    //
    //
    //    //Q_LENGTH-1 because no center item
    //    for (int item_index=0; item_index<Q_LENGTH-1; item_index++){
    //
    //        int item = active_items[item_index];
    //
    //
    //
    //        tNi x0 = node.xg0;
    //        tNi y0 = node.yg0;
    //        tNi z0 = node.zg0;
    //
    //
    //        tNi x1 = node.xg1;
    //        tNi y1 = node.yg1;
    //
    //
    //
    //        tNi zlen = node.z;
    //
    //
    //
    //        tNi start_end_length[27][7] = {
    //
    //            { 0,  0,  0,  0,  0,  0,    1},
    //            { 1, x1,  0,  0,  0,  0,    1},
    //            {x0, x0,  0,  0,  0,  0,    1},
    //
    //            { 0,  0,  1, y1,  0,  0,    1},
    //            { 1, x1,  1, y1,  0,  0,    1},
    //            {x0, x0,  1, y1,  0,  0,    1},
    //
    //            { 0,  0, y0, y0,  0,  0,    1},
    //            { 1, x1, y0, y0,  0,  0,    1},
    //            {x0, x0, y0, y0,  0,  0,    1},
    //
    //
    //            { 0,  0,  0,  0,  1,  1, zlen},   //Item  9
    //            { 1, x1,  0,  0,  1,  1, zlen},   //Item 10 top
    //            {x0, x0,  0,  0,  1,  1, zlen},   //Item 11
    //
    //            { 0,  0,  1, y1,  1,  1, zlen},   //Item 12
    //            {109,108,107,106,105,104,   0},   //Center not used
    //            {x0, x0,  1, y1,  1,  1, zlen},   //Item 14
    //
    //            { 0,  0, y0, y0,  1,  1, zlen},   //Item 15
    //            { 1, x1, y0, y0,  1,  1, zlen},   //Item 16 bottom
    //            {x0, x0, y0, y0,  1,  1, zlen},   //Item 17
    //
    //
    //            { 0,  0,  0,  0, z0, z0,    1},
    //            { 1, x1,  0,  0, z0, z0,    1},
    //            {x0, x0,  0,  0, z0, z0,    1},
    //
    //            { 0,  0,  1, y1, z0, z0,    1},
    //            { 1,  x1, 1, y1, z0, z0,    1},  //Item 22
    //            {x0,  x0, 1, y1, z0, z0,    1},
    //
    //            { 0,  0, y0, y0, z0, z0,    1},
    //            { 1, x1, y0, y0, z0, z0,    1},
    //            {x0, x0, y0, y0, z0, z0,    1}
    //        };
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //        //LOOPING THROUGH CELLS (NOT VECTORS)
    //
    //        for (tNi i_dst = start_end_length[item][0]; i_dst <= start_end_length[item][1]; i_dst++){
    //            for (tNi j_dst = start_end_length[item][2]; j_dst <= start_end_length[item][3]; j_dst++){
    //                for (tNi k_dst = start_end_length[item][4]; k_dst <= start_end_length[item][5]; k_dst++){
    //
    //
    //
    //                    tNi i_src = i_dst + item_to_cell_dir[ item ][0];
    //                    tNi j_src = j_dst + item_to_cell_dir[ item ][1];
    //                    tNi k_src = k_dst + item_to_cell_dir[ item ][2];
    //
    //
    //
    //                    tNi k_START = k_dst;
    //
    //                    tNi num_cells = start_end_length[ item ][6];
    //
    //
    //                    for (tNi zcell=0; zcell<num_cells; zcell++){
    //
    //
    //                        k_dst = k_START + zcell;
    //
    //
    //
    //                        //for vector in item
    //                        for (int index=0; index<num_vectors_in_item[ item ]; index++){
    //
    //                            int v = vectors_in_item_array[ item ][ index ];
    //
    //                            int l_src = vector_reverse_map[v];
    //
    //                            int l_dst = vector_map[v];
    //
    //
    //
    //
    //                            //DO THE REVERSE
    //                            //example, i,j,k  1,0,0, and item 1, and vector 1.  (l=xx)
    //                            i_src = i_dst + VectorDir[v][0] * -1;
    //                            j_src = j_dst + VectorDir[v][1] * -1;
    //                            k_src = k_dst + VectorDir[v][2] * -1;
    //
    //
    //                            int d_src = (int)(i_src - 1) / device.x_fraction;
    //                            int d_dst = (int)(i_dst - 1) / device.x_fraction;
    //
    //                            tNi pos_src = ((i_src- 1) % device.x_fraction + 1);
    //                            tNi pos_dst = ((i_dst - 1) % device.x_fraction + 1);
    //
    //
    //
    //
    //
    //                            //Reverse is always on the HALOS
    //                            N[d_dst][ pos_dst + l_dst] = N[d_src][ pos_src + l_src];
    //
    //
    //
    //
    //
    //                        }//end of vectors in item
    //
    //
    //                    }//end of zcell
    //
    //
    //
    //                }}}//end of for i, j, k
    //
    //
    //
    //    }//end of item loop
    //
    //
    //
    //
    //}//end of func



    
}
