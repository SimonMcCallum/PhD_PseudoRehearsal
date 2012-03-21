//                              -*- Mode: C -*- 
// learning.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: learning.h,v 1.6 2003/06/16 07:39:59 simon Exp $
// 


#ifndef _learning_h_
#define _learning_h_


int createSymmetry(netinfo *net);

void blurPattInputs(netinfo *net,traininginfo * training, pattern * pat, double ratio);
double calcPattErrors(netinfo *net,traininginfo * training,pattern * desired, pattern *actual);

int updateWeightDeltas(netinfo *net,traininginfo * training,pattern * pat, double ratio);
double updateWeightDeltasNoise(netinfo * net, traininginfo * training, pattern * inputPat, BOOL PseudoPatt);
double updateWeightDeltasNoNoise(netinfo * net, traininginfo * training, pattern * inputPat, BOOL PseudoPatt);

int hebbianLearnAll(netinfo * net, traininginfo * training);
int hebbianLearnPartial(netinfo * net, traininginfo * training, pattern * patternArray, int * patternOrder,  int start,int end, double learningConst);

int calcPattInputs(netinfo *net,traininginfo * training,pattern * pat);
double calcPattDeltas(netinfo *net,traininginfo * training,pattern * pat);
int updateNetWeights(netinfo *net,traininginfo * training,pattern * pat);

double deltaLearnAll(netinfo * net, traininginfo * training);
double deltaLearnPartial(netinfo * net, traininginfo * training, int start, int end);

/*
 Test a new pattern against all learnt patterns
 returns the position of the pattern in the list of patterns to learn
 or NOTFOUND (-1) if not present
 */
int compareToLearntPatterns(netinfo * net, traininginfo * training, pattern * setPat);

#endif
