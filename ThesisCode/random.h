/*
	File:	Random.h
	Author:	Simon McCallum
	Description:
		Random functions that are useful that are not provided by standard
		rnadom libraries.
		Also used so that different OS's can provide the same random functions
        
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TRUE  1
#define FALSE 0

//#define MACOSX

//extern rand();


void   RandomInitialise(int,int);
double RandomUniform(void);
double RandomGaussian(double,double);
int    RandomInt(int,int);
double RandomDouble(double,double);




void seedRandom(long seedValue);
long initializeRandomSeed();
int  nextRandomInt();
double nextRandomDouble();


