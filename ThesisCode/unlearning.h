//                              -*- Mode: C -*- 
// unlearning.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: unlearning.h,v 1.3 2003/05/06 09:37:17 simon Exp $
// 


#ifndef _unlearning_h_
#define _unlearning_h_

int addUnlearningItem(netinfo * net, traininginfo * training, pattern * setPat);
int compareToUnlearningPatterns(netinfo * net, traininginfo * training, pattern * setPat);

int generateUnlearningPatterns(netinfo * net, traininginfo * training);


/*
* This function is designed to remove the large basins of attraction in the network
* It does this by randomly sampling the network to find the large basin.  If one network is 
dominating the network, as it the case with overloading, then the stable pattern found will be 
this large dominant pattern.  The standard algorithm is to unlearn this stable pattern to some extent. 

*
*/
int unlearnCycles(netinfo * net, traininginfo * training,int cycles);

#endif
