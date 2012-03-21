//                              -*- Mode: C -*- 
// general.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: general.h,v 1.6 2003/10/21 11:24:03 simon Exp $
// 

#include "random.h"

#ifndef _gen_h_
#define _gen_h_

//used for comparison for equality of real numbers
#define SMUDGE 0.000000000000001
#define MAX_STRING_LENGTH 200
// a smaller way to write for loops
#define FOR(x,y) for(x=0;((x)<(y));x++)
#define SG(x)(((x)>0)?(1):(-1))
#define SameSign(i,w) (((i<0)&&(w<0))||((i>0)&&(w>0)))? fabs(w) : 0
#define DiffSign(i,w) (((i>0)&&(w<0))||((i<0)&&(w>0)))? fabs(w) : 0

#define MAX(x,y) ((x)>(y))?(x):(y)
#define MIN(x,y) ((x)<(y))?(x):(y)

#define XOR(x,y) (((x)&&!(y))||(!(x)&&(y)))

#define SQUARE(x) ((x)*(x))
#define sqr(x) ((x)*(x))
#define SQR(x) ((x)*(x))

#define BOOL int

int withinMargin(double a, double b);
void swap(int * left, int * right);

void randomizeArray(int * array1, int size1);

int * randomizeOrder(int * array1, int size1);



#endif
