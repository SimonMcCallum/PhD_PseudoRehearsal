//                              -*- Mode: C -*- 
// sampling.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: sampling.h,v 1.4 2003/05/11 20:59:01 simon Exp $
// 


#ifndef _sampling_h_
#define _sampling_h_


int addStablePat(netinfo * net, traininginfo * training, pattern * pat, int cycleDist, int patClose, int patDist, int patInitChanges);

/*Returns the overlap for which 50% of samples relaxs to the actual pattern*/ 
double sampleBasinOfAttraction(netinfo * net, traininginfo * training, pattern * attractor);


int sortAcrossMultiplePatterns(netinfo * net, traininginfo * training, int * order);

int sampleNet(netinfo * net, traininginfo * training);

int checkAgainstLearntPatterns(netinfo * net, traininginfo * training, int cycleDist, int * patClose, int * patDist, int patInitChanges);

int checkAgainstOtherPatterns(netinfo * net, traininginfo * training, int cycleDist, int initialChanges);

#endif
