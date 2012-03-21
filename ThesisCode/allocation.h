//                              -*- Mode: C -*- 
// allocation.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: allocation.h,v 1.4 2002/12/08 13:36:11 simon Exp $
// 


#ifndef _allocation_h_
#define _allocation_h_
#include "training.h"

pattern * allocPattern(netinfo * net, pattern*pat);
pattern * initPattern(netinfo * net, pattern*pat);
void freePattern(pattern*pat);


/********************************************************************************************************
*/
int allocPatternMemory(netinfo * net, traininginfo * training);

/********************************************************************************************************
*/
int allocNetMemory(netinfo * net);

#endif
