//                              -*- Mode: C -*- 
// net.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: net.h,v 1.10 2003/05/17 14:01:28 simon Exp $
// 


#ifndef _net_h_
#define _net_h_

#define E 2.718281828459045

#define FOUND 0
#define NOTFOUND -1
#define NUMSTABLEFULL -2

enum {THRESHOLD,SIGMOID};
enum {HEBBIAN, PSEUDODELTA, STRICTDELTA, UNLEARNING};  //learning types
enum {BATCH, ONLINE};
enum {SYMMETRIC, ASYMMETRIC};
enum {ASYNCHRONOUS, SYNCHRONOUS};
enum {NOCAP, HARDCAP, SOFTCAP};
enum {NONORMALISATION, NORMALISENET, NORMALISEUNIT, NORMALISEUNIT_ZEROMEAN, NORMALISEUNIT_BOTH};
enum {ABSOLUTE , RELATIVE};
enum {DELTAERROR, SIGMOIDERROR};
enum {learnt,spurious};
//
//  Constants and macros:
//

//
//  Type declarations:
//

struct pattern_type
  {
    int num;
    double * unit;
    double * unitInput;
    double * unitNetInput;
    double * unitDelta;
    double * unitError;
    int    sum; //of units on
    int    cycleDist;
    BOOL   cycle;
    BOOL   stable;
    double leastStable;
    double meanStable;
    double mostStable;
    double stddev;
    double coefficientOfVariation;
    double ratioStable;
    double basinSize;
    double meanUnitInput;

    int hammingDist;
    int frequency;
    int minimumDistToLearntPattern;
    int closestLearntPattern;
	int initialChanges;
    
};
typedef struct pattern_type pattern;



struct netinfo_type
  {
    int N;
    pattern pat; //This is the current pattern in the Network.

    pattern initPat;

    pattern noisyPat;

    double * noisyInput;
    int * unitsToChange;
    int * unitOrder;

    double unitsInRatio; 

    double * weightArray;
    double ** weight;
    double * weightDeltaArray;
    double ** weightDelta;


    double ON;
    double OFF;
    int  maxCycles;
    int  activationType;
    BOOL bias;    
    int  learnTiming;
    int  relaxation;
    BOOL symmetricWeights;

	BOOL thermalDelta;
	double startTemperature;
	double temperature;

    BOOL noiseDuringRelaxation;
    BOOL calcHammingDistRelax;

	BOOL calcInitialChanges;
	int initialChanges;

	BOOL calcWeightAverages;

    int gaussianNoiseType;
    int lenghtOfTimeStableInRelaxation;
    double gaussianRelativeRangeForRelaxation;
    double gaussianAbsoluteRangeForRelaxation;
    
    int weightCappingType; // choises are NOCAP, HARDCAP, SOFTCAP
    double weightCappingValue; // the point at which to cap weights;
    
    BOOL weightDecay; // True of false; 
    double weightDecayValue; //10% ;

    int weightNormalisationType; // choises are NONORMALISATION,NORMAILSENET,NORMALISEUNIT
    double weightNormalisationValue; // the point at which to cap weights;

    long randomSeed; /*This is saved so that we can rerun simulations. if it is present then it will be used.  If not a new random number will be created and saved with the param file.*/
    
    double sampleProbOn; //The probability of a unit begin on during random sampling of patterns
    //must check with generateProbOn in training.  
 
	double averageInput; // this is used to work out if you are higher than average - calculated as a rolling average of 10, only counted for stable units.

    BOOL gnuplotGraphs;

    int trial;
    
    int debugLevel;
  };

typedef struct netinfo_type netinfo;



int initWeights(netinfo * net);

int decayWeights(netinfo * net);

double addToWeight(netinfo * net, double * weight, double value);
double normaliseWeights(netinfo * net);

/*
functions to reset arrays to 0
*/
void resetDeltas(netinfo *net);
void resetErrors(netinfo *net);
void resetWeightDeltas(netinfo *net);

/*
Activation functions
*/
double activation(netinfo * net, double input);
double activationSigmoid(netinfo * net, double input);

int setNetworkRandom(netinfo * net);

double avgWeight(netinfo * net);


int calcInputs(netinfo * net);
int calcOutputs(netinfo * net);
void blurNetInputs(netinfo *net, double * inputs);
int relaxNetworkSynchronous(netinfo * net);

int findNumToChange(netinfo * net, int * unitsToChange);
int findNumToChangeNoise(netinfo * net, int * unitsToChange, double * noisyInput);
int updateInputsUnitChanged(netinfo * net, int unit, double oldActivation);
int relaxNetworkAsynchronous(netinfo * net);

//NON-Distructive checking of pattern stability
int stablePattern(netinfo *net, pattern *pat);

int relaxNetwork(netinfo * net);

int numcmp(void const * a, void const * b);

double sortPatternInput(netinfo * net, pattern * pat);


int setState(netinfo * net, pattern * to, pattern * from);

int setNetworkPartial(netinfo * net, pattern * pat, double overlap);

int unstable(netinfo * net);
int numUnstable(netinfo * net);


/*************************************************************************
This does equal to both real pattern and the inverse of the left pattern
Returns Hamming distance*/
int equalState(netinfo * net, pattern * left, pattern * right);

int fastEqualState(netinfo * net, pattern * left, pattern * right);

int copyPattern(netinfo * net, pattern * toPat,pattern * fromPat);




#endif
