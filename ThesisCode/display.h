//                              -*- Mode: C -*- 
// display.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: display.h,v 1.7 2003/12/14 15:28:03 simon Exp $
// 


#ifndef _display_h_
#define _display_h_

void printNetwork(netinfo * net);

void printNetInput(netinfo * net);

void printWeights(netinfo * net);
void printWeightsSummary(netinfo * net);

void printNetInputSummary(netinfo * net);

void displayPatt(netinfo * net, pattern * pat,int * order);
void displayPattInput(netinfo * net, pattern * pat,int * order);
void displayPattNetInput(netinfo * net, pattern * pat);
void displayPattSummary(netinfo * net,traininginfo * training, pattern * pat, int learning);

void displayPatternResults(netinfo * net, traininginfo * training,int * order);

void displayRatioResultsSummary(netinfo * net, traininginfo * training, singleRun * multiTrialSummary);

void displayFinalSummaryStabilityofPosition(netinfo * net, traininginfo * training,singleRun *a);
void displayFinalSummaryDistanceTraveled(netinfo * net, traininginfo * training,singleRun *a);
void displayFinalSummaryStabilityofPositionBasin(netinfo * net, traininginfo * training,singleRun *a);
void displayFinalSummaryStabilityofPositionBasinError(netinfo * net, traininginfo * training,singleRun *a);

void displayFinalSummary(netinfo * net, traininginfo * training,singleRun *a);
void initRun(singleRun * a);
void calcValuesRun(netinfo * net, traininginfo * training, singleRun * a);
void displayValuesRun(netinfo * net, traininginfo * training,singleRun *a);
void addValuesSummary( singleRun * multiTrialSummary, singleRun * a);


void displayAvgStabProfile(netinfo * net, traininginfo * training);

void saveStabilityProfilePattern(netinfo * net, traininginfo * training, FILE * stabilityProfileFile, int pat);
void saveStabilityProfileSpuriousPattern(netinfo * net, traininginfo * training, FILE * stabilityProfileFile, int pat);
void saveSaturationProfilePattern(netinfo * net, traininginfo * training, FILE * saturationProfileFile, int pat);
void saveSaturationProfileSpuriousPattern(netinfo * net, traininginfo * training, FILE * saturationProfileFile, int pat);
#endif
