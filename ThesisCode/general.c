#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "general.h"
#include "random.h"


/********************************************************************************************************
return 1 if the two numbers are within Smudge of each other
*/
int withinMargin(double a,double b){
    return (((a>(b-SMUDGE))&&(a<(b+SMUDGE)))|| a==b);
}

/***********************************************
Standard pointer based swap
*/
void swap(int * left, int * right){
    int  temp;
    temp = *right;
    *right = *left;
    *left = temp;
}

/****************************************
This function is used to randomise the numbers in an array passed in as array1.
In:
    <int> The length of the array
    <int *> The array to rearrange 
Out:
    The changes to the array pointed to by array1.
Calls:
    swap(i,j) inside loop.

Uses swap(i,j) moving i up array with random j position later in the array
*/
void randomizeArray(int * array1,int size1)
{
  int i,pos;
  for(i=0;i<size1;i++)
    {
      pos = (size1-1) - nextRandomInt()%(size1-i);
      swap(array1+pos,array1+i);
    }
}



/****************************************
This function is used to allocate memory for an array, fill it with numbers, and then randomise the order of those numbers
In:
    <int> The length of the array
Out:
    <int *> The pointer to the array that has been allocated, and filled with random numbers.
Calls:
    randomiseArray(i,j), called Once.
*/
int * randomizeOrder(int * array1, int size1)
{
  int i;
  if (array1==NULL){
    array1 = calloc(sizeof(int),size1);
  }
  for(i=0;i<size1;i++)
    {
      array1[i]=i;
    }
  randomizeArray(array1,size1);
  return array1;
}

