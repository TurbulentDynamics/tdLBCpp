//
//  Header.h
//  tdLBcpp
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#if !defined(WITH_GPU) && !defined(WITH_GPU_MEMSHARED)
#define WITH_CPU
#define HOST_DEVICE_GPU
#else
#define HOST_DEVICE_GPU __host__ __device__
#endif

//Debug trace
#ifdef DEBUG
#define LOG(args...) printf(args)
#else
#define LOG(ignored...)
#endif


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
//y positive up
//z positive backwards


//Classical Q27 vector names, x, y, z


//The Zeroth direction is the central non-moving direction.  So all
//vectors are numbered from 1.  The faces are enumerated first,
//then the edges, then corners.

//Q0 unused, so not allocated in memory, and Q19, acually
#define Q0 Unused_Error    // 0,  0,  0



//Faces
#define Q01  0         //+1,  0,  0  //RIGHT
#define Q02  1         //-1,  0,  0
#define Q03  2         // 0, +1,  0  //UP
#define Q04  3         // 0, -1,  0
#define Q05  4         // 0,  0, +1  //BACK
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




#define CENTER UNDEFINED

//Faces
#define RIGHT Q01
#define LEFT Q02
#define UP Q03
#define DOWN Q04
#define BACK Q05
#define FORWARD Q06

//Edges
#define RIGHTUP Q07
#define LEFTDOWN Q08
#define RIGHTBACK Q09
#define LEFTFORWARD Q10
#define UPBACK Q11
#define DOWNFORWARD Q12
#define RIGHTDOWN Q13
#define LEFTUP Q14
#define RIGHTFORWARD Q15
#define LEFTBACK Q16
#define UPFORWARD Q17
#define DOWNBACK Q18

//Corners
#define RIGHTUPBACK Q19
#define LEFTDOWNFORWARD Q20
#define RIGHTUPFORWARD Q21
#define LEFTDOWNBACK Q22
#define RIGHTDOWNBACK Q23
#define LEFTUPFORWARD Q24
#define LEFTUPBACK Q25
#define RIGHTDOWNFORWARD Q26


//Ortho Q27 names, counting from top to the right then backwards
#define O00 Q24
#define O01 Q17
#define O02 Q21

#define O03 Q10
#define O04 Q06
#define O05 Q15

#define O06 Q20
#define O07 Q12
#define O08 Q26

#define O09 Q14
#define O10 Q03
#define O11 Q07
#define O12 Q02
#define O13 CENTER
#define O14 Q01
#define O15 Q08
#define O16 Q04
#define O17 Q13

#define O18 Q19
#define O19 Q11
#define O20 Q19

#define O21 Q16
#define O22 Q05
#define O23 Q09

#define O24 Q22
#define O25 Q18
#define O26 Q23


//Collections of Vectors
#define nFACES 6
#define FACES [Q01, Q02, Q03, Q04, Q05, Q06]

#define nEDGES 12
#define EDGES [Q07, Q08, Q09, Q10,   Q11, Q12, Q13, Q14,   Q15, Q16, Q17, Q18]

#define nCORNERS 8
#define CORNERS [Q19, Q20, Q21, Q22, Q23, Q24, Q25, Q26]



#define nRIGHT 9
#define RIGHTS [O02, O05, O08, O11, O14, O17, O20, O23, Q26]

#define nLEFT 9
#define LEFTS [O00, O03, O06, O09, O12, O15, O18, O21, Q24]

#define nUP 9
#define UPS [O00, O01, O02, O09, O10, O11, O18, O19, O20]

#define nDOWN 9
#define DOWNS [O06, O07, O08, O15, O16, O17, O24, O25, O26]

#define nBACK 9
#define BACKS [O18, O19, O20, O21, O22, O23, O24, O25, O26]

#define nFORWARD 9
#define FORWARDS [O01, O02, O03, O04, O05, O06, O07, O08, O09]
