//
//  Header.h
//  tdLBcpp
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once


#define WITH_CPU


//Dummies when there is no mpi.h included
#if WITH_MPI == 0
struct MPI_Group{};
struct MPI_Win{};
struct MPI_Comm{};
struct MPI_Barrier{};
#define MPI_COMM_WORLD
#endif

enum QLen {D3Q19 = 18, D3Q27 = 26};
enum Collision {EgglesSomers, EgglesSomersLES, Entropic};
enum Streaming {Simple, Esotwist};
enum MemoryLayoutType {MemoryLayoutIJKL, MemoryLayoutLIJK};


using tNi = long int;
using let_tNi = const long int;


using tStep = unsigned long int;
using let_tStep = const unsigned long int;



//Coordinate system
//x positive to right
//y
//z


//Classical Q27 vector names, x, y, z


//The Zeroth direction is the central non-moving direction.  So all
//vectors are numbered from 1.  The faces are enumerated first,
//then the edges, then corners.

//Q0 unused, so not allocated in memory, and Q19, acually
#define Q0 Unused_Error    // 0,  0,  0



//Faces
#define Q01  0         //+1,  0,  0
#define Q02  1         //-1,  0,  0
#define Q03  2         // 0, +1,  0  //UP VECTOR
#define Q04  3         // 0, -1,  0  //DOWN VECTOR
#define Q05  4         // 0,  0, +1
#define Q06  5         // 0,  0, -1

//Edges
#define Q07  6         //+1, +1,  0
#define Q08  7         //-1, -1,  0
#define Q09  8         //+1,  0, +1
#define Q10  9         //-1,  0, -1
#define Q11 10         // 0, +1, +1
#define Q12 11         // 0, -1, -1
#define Q13 12         //+1, -1,  0
#define Q14 13         //-1, +1,  0
#define Q15 14         //+1,  0, -1
#define Q16 15         //-1,  0, +1
#define Q17 16         // 0, +1, -1
#define Q18 17         // 0, -1, +1

//Corners
#define Q19 18         //+1, +1, +1
#define Q20 19         //-1, -1, -1
#define Q21 20         //+1, +1, -1
#define Q22 21         //-1, -1, +1
#define Q23 22         //+1, -1, +1
#define Q24 23         //-1, +1, -1
#define Q25 24         //-1, +1, +1
#define Q26 25         //+1, -1, -1



#define MRHO 0
#define M01  0
#define M02  1
#define M03  2
#define M04  3
#define M05  4
#define M06  5
#define M07  6
#define M08  7
#define M09  8
#define M10  9
#define M11 10
#define M12 11
#define M13 12
#define M14 13
#define M15 14
#define M16 15
#define M17 16
#define M18 17
#define M19 18
#define M20 19
#define M21 20
#define M22 21
#define M23 22
#define M24 23
#define M25 24
#define M26 25











#define CENTER Unused


//Faces
//#define RIGHT Q01
//#define LEFT Q02
//#define UP Q03
//#define DOWN Q04
//#define BACK Q05
//#define FORWARD Q06
//
////Edges
//#define RIGHTDOWN Q07
//#define LEFTUP Q08
//#define RIGHTBACK Q09
//#define LEFTFORWARD Q10
//#define DOWNBACK Q11
//#define UPFORWARD Q12
//#define RIGHTUP Q13
//#define LEFTDOWN Q14
//#define RIGHTFORWARD Q15
//#define LEFTBACK Q16
//#define DOWNFORWARD Q17
//#define UPBACK Q18
//
////Corners
//#define RIGHTDOWNBACK Q19
//#define LEFTUPFORWARD Q20
//#define RIGHTDOWNFORWARD Q21
//#define LEFTUPBACK Q22
//#define RIGHTUPBACK Q23
//#define LEFTDOWNFORWARD Q24
//#define LEFTDOWNBACK Q25
//#define RIGHTUPFORWARD Q26



//Ortho Q27 names, counting from top to the right then backwards
// FIXME :
//#define O0 Q20
//#define O1 Q12
//#define O2 Q26
//#define O3 Q10
//#define O4 Q06
//#define O5 Q15
//#define O6 Q24
//#define O7 Q17
//#define O8 Q21
//
//#define O9 Q08
//#define O10 Q03
//#define O11 Q13
//#define O12 Q02
////#define O13 Q0
//#define O14 Q01
//#define O15 Q14
//#define O16 Q04
//#define O17 Q07
//
//#define O18 Q22
//#define O19 Q18
//#define O20 Q23
//#define O21 Q16
//#define O22 Q05
//#define O23 Q09
//#define O24 Q25
//#define O25 Q11
//#define O26 Q19
//
//#define nFACES 6
//#define FACES [O04, O10, O12, O14, O16, O22]
//
//#define nEDGES 12
//#define EDGES [O01, O03, O05, O07,   O09, O11, O15, O17,   O19, O21, O23, O25]
//
//#define nCORNERS 8
//#define CORNERS [O00, O02, O06, O8, O18, O20, O24, O26]
//
//#define nTOP 9
//#define TOP [O00, O01, O02, O09, O10, O11, O18, O19, O20]
