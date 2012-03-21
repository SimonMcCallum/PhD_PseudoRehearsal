//                              -*- Mode: C -*- 
// patternFileIO.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: patternFileIO.h,v 1.3 2003/01/02 11:27:30 simon Exp $
// 


#ifndef _patternFileIO_h_
#define _patternFileIO_h_

void saveWeights(netinfo * net);
void loadWeights(netinfo * net);

void savePatt(netinfo * net, FILE * patternFile, pattern * pat,int * order);
void saveAllLearntPatterns(netinfo * net, traininginfo * train,int * order);

void loadPatt(netinfo * net,  FILE * patternFile, pattern * pat, int numSpare);
void loadAllLearntPatterns(netinfo * net, traininginfo * train);

#endif
