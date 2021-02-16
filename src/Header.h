//
//  Header.h
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

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


using tNi = long int;
using let_tNi = const long int;


using tStep = int;
using let_tStep = const int;



//Coordinate system
//x positive to right
//y positive downwards
//z positive backwards


//Classical Q27 vector names, x, y, z


//The Zeroth direction is the central non-moving direction.  So all
//vectors are numbered from 1.  The faces are enumerated first,
//then the edges, then corners.

//Q0 unused, so not allocated in memory, and Q19, acually
#define Q0 Unused_Error    // 0,  0,  0

//Faces
#define Q1 0           //+1,  0,  0
#define Q2 1           //-1,  0,  0
#define Q3 2           // 0, +1,  0
#define Q4 3           // 0, -1,  0
#define Q5 4           // 0,  0, +1
#define Q6 5           // 0,  0, -1

//Edges
#define Q7 6           //+1, +1,  0
#define Q8 7           //-1, -1,  0
#define Q9 8           //+1,  0, +1
#define Q10 9          //-1,  0, -1
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




//#define CENTER Unused


//Faces
#define RIGHT Q1
#define LEFT Q2
#define UP Q3
#define DOWN Q4
#define BACK Q5
#define FORWARD Q6

//Edges
#define RIGHTDOWN Q7
#define LEFTUP Q8
#define RIGHTBACK Q9
#define LEFTFORWARD Q10
#define DOWNBACK Q11
#define UPFORWARD Q12
#define RIGHTUP Q13
#define LEFTDOWN Q14
#define RIGHTFORWARD Q15
#define LEFTBACK Q16
#define DOWNFORWARD Q17
#define UPBACK Q18

//Corners
#define RIGHTDOWNBACK Q19
#define LEFTUPFORWARD Q20
#define RIGHTDOWNFORWARD Q21
#define LEFTUPBACK Q22
#define RIGHTUPBACK Q23
#define LEFTDOWNFORWARD Q24
#define LEFTDOWNBACK Q25
#define RIGHTUPFORWARD Q26



//Ortho Q27 names, counting from top to the right then backwards
#define O0 Q20
#define O1 Q12
#define O2 Q26
#define O3 Q10
#define O4 Q6
#define O5 Q15
#define O6 Q24
#define O7 Q17
#define O8 Q21

#define O9 Q8
#define O10 Q3
#define O11 Q13
#define O12 Q2
//#define O13 Q0
#define O14 Q1
#define O15 Q14
#define O16 Q4
#define O17 Q7

#define O18 Q22
#define O19 Q18
#define O20 Q23
#define O21 Q16
#define O22 Q5
#define O23 Q9
#define O24 Q25
#define O25 Q11
#define O26 Q19


#define FACES [O4, O10, O12, O14, O16, O22]
#define EDGES [O1, O3, O5, O7,   O9, O11, O15, O17,   O19, O21, O23, O25]
#define CORNERS [O0, O2, O6, O8, O18, O20, O24, O26]
