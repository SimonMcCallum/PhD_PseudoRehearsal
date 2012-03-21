//                              -*- Mode: C -*- 
// training.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: training.h,v 1.15 2003/12/14 15:28:03 simon Exp $
// 


#ifndef _training_h_
#define _training_h_
enum {unlearningStandard,unlearningSpuriousRatio,unlearningSpuriousMean,unlearningSpuriousOnly};

struct singleRunType
{
  int numSpuriousPatts;
  int numLearntPatts;
  double maxFequencySpurious;
  double minFequencyLearnt;
  double maxRatioSpurious;
  double minRatioLearnt;
  double minBasinSpurious;
  double maxBasinLearnt;
  double maxStdVarLearnt;
  double minStdVarSpurious;
  double maxMeanSpurious;
  double minMeanLearnt;
  double maxLowSpurious;
  double minLowLearnt;
  double maxCycleDist;
  int misClassifiedRatio;
  int misClassifiedLeastStable;
  int misClassifiedRatioSet;
  int classifiedRatioSet;
  int misClassifiedStdVar;
  int misClassifiedStdVarSet;
  int classifiedStdVarSet;
  int misClassifiedMean;
  int misClassifiedNum;
  int PPVRatio;
  int NPVRatio;
  int TNRRatio;
  int TPRRatio;
  int PPVMean;
  int NPVMean;
  int TNRMean;
  int TPRMean;
  int PPVNum;
  int NPVNum;
  int TNRNum;
  int TPRNum;
  int RatioAStdVarDisagree;
  int numLearntUnstable;
  int numLearntStable;
  long numLearntStableSqr;
  int numBPStable;
  long numBPStableSqr;
  int numLearntFound;
  long sumLearntFrequency;
  // long sumCircling; //again not used
  long sumSpuriousFrequency;
  double * patternsStabilityOverManyTrials;
  double * patternsStabilityOverManyTrialsSqr;
  double * patternsDistOverManyTrials;
  double * patternsStabilityBasinOverManyTrials;
  double * patternsStabilityBasinSumSqrOverManyTrials;
};

typedef struct singleRunType singleRun;

struct traininginfo_type
{
  int numTrials; // This determines how many trial are run
  
  double learningConst;
  double learningConstDecay;
  double learningConstDecayTo;
  double momentum;


  //Variables set by user:
  int maxLearntPatts; // The number of patterns to learn generally set at the max value with all cycles
  BOOL patternsFromFile;
  char patternFilename[MAX_STRING_LENGTH];


  int numCheck;       // The sampling size when checking the network
  int maxSpuriousPatts; //maximum number of patterns to store as Stable Spurious pattern
  int maxPseudoPatts; //maximum number of patterns to store as Pseudo patterns
  int maxSamplesForPseudoPatts; //How many sample to try to get the appropriate number of pseudo items
  int maxUnlearningPatts; //maximum number of patterns to store as Pseudo patterns
  int maxSamplesForUnlearningPatts; //How many sample to try to get the appropriate number

  double pseudoRehearsalCutoff; // this is the cutoff for inclusion in the pseudo population.

  BOOL realRehearsal;
  double realRehearsalPercent;

  BOOL includePseudoInError;
  BOOL noiseOnPseudoPatts;
  double noiseRatioPseudoPatts;
  double learningRatioPseudoPatts;


  BOOL useMaxLowestLearntvsCutoff;
  double minLearntPatternRatio;  // this is the current lowest ratio for a learnt pattern


  double unlearningCutoffHigh;
  double unlearningCutoffLow;
  int unlearningType;
  BOOL unlearningSuperHeated;
  int unlearningCycleSize;

  double unlearningConst; // the amount to unlearn for each unlearning pattern
  
  int trainingEpochs;    //number of epoachs to train
  int initialTrainingEpochs;    //number of epoachs to train on the first pass
  BOOL allEpochLearning; //do not stop early - run all epochs
  double errorCriteria;  //error criteria while traing
  double errorTail;      //decay of the error value creating an error tail

  int hebbianCycles;  // number of cycles used for hebbian learning
  double hebbianNoiseLevel; // noise level while training


  //Binary choice modal types:
  BOOL noiseHebbianNegative;
  //This is the parameter that deterimines what the noise during hebbian learning is actually goint to do.

  //Modal types:
  int learningType;   //one of STRICT DELTA, HEBBIAN, PSEUDODELTA
  int errorCalcType; //SIGMOIDERROR , DELTAERROR the type of error.

  
  //Instance Variables
  int numSpuriousPatts;	// varies to show that actual number of patterns are stable currently
  int numPseudoPatts; //the current number of PseuoPatts
  int numUnlearningPatts; //the current number of PseuoPatts

  int step;
  int initialNumberPatts;
  int numLearntPatts;

  
  BOOL uniquePseudoPatts;  // this makes sure that a pseudo item only appears once in an epoach
  BOOL pseudoRehearsalReal; // rehearse only the learnt patterns that where found
  BOOL removeLearntFromPseudoPatts; // this makes sure that the learnt patterns are not part of the pseudo patterns.  This is not a useful thing to do except during testing to see if PR actually works with this restriction.

  BOOL uniqueUnlearningPatts;  // this makes sure that a unlearning item only appears once in an epoch
  BOOL removeLearntFromUnlearningPatts; // this makes sure that the learnt patterns are not part of the unlearning patterns.  This would allow unlearning to work much betteras you would not remove the actual learning pattern.
    
  BOOL adjustLearningForPobability;
    
  BOOL noiseOnInput; //T or F noise on the input of units during training
  BOOL noiseOnOutput; // T or F noise on the output of units during training
  BOOL noiseHetroAssociative; // T or F map a noisy pattern to a real pattern.

  double hetroNoiseLevel; //percent chance of unit flipping

  double gaussianRelativeRange;
  double gaussianAbsoluteRange;
  int gaussianNoiseType; // ABSOLUTE , RELATIVE (to the mean absolute input value)

    
  double generateProbOn; //The probability of a unit begin on in the random generation of patterns
  int generateActivationExactLevel;

  BOOL calcSpuriousBasins;
  BOOL calcLearntBasins;

  BOOL displaySpuriousPatts;
  BOOL displaySpuriousPatternDetails;

  BOOL reorderUnitsinPatts; /* this is the param that tells us to
			       reorder the units in the patterns so that they follow the pattern
			       ########........ then ####....####.... then ##..##..##..##.. etc*/
 
    

  BOOL  saveStabilityProfilePatternCheck;
  int * saveStabilityProfilePattern; 
  int numSpuriousPatsProfiles;
            
  //Display check boxes
  BOOL displayTestedPatterns;

  BOOL displayLearntPattUnits;
  BOOL displayLearntPattInput;
  BOOL displayLearntPattNetInput;
  BOOL displayLearntPattSummary;

  BOOL displaySpuriousPattUnits;
  BOOL displaySpuriousPattInput;
  BOOL displaySpuriousPattNetInput;
  BOOL displaySpuriousPattSummary;

  BOOL displayPseudoPattUnits;
  BOOL displayPseudoPattInput;
  BOOL displayPseudoPattNetInput;
  BOOL displayPseudoPattSummary;

    
    
  singleRun * allRuns; //this is for storing useful data
    
  int * learntOrder; 
  int * fixedOrder;
  pattern * learntPatts; //the Patterns that are to be learnt
  pattern * spuriousPatts; //Patterns that are stable
  pattern * pseudoPatts; //Storage for the pseudo patterns, used in PR.
  pattern * unlearningPatts; //Storage for the unlearning patterns, used in .
};

typedef struct traininginfo_type traininginfo;

int setNetwork(netinfo * net, traininginfo * training, pattern * patternSet);

int generateUnitInputsIndepenantActivation(netinfo * net, traininginfo * training);
int generateUnitInputsExactActivationLevel(netinfo * net, traininginfo * training);
int generateUnitInputs(netinfo * net, traininginfo * training);

int generatePatterns(netinfo * net, traininginfo * training);

int initAll(netinfo * net, traininginfo * training);

void knuthShuffle(int * callSequence,int size);
void initSequence(int * callSequence,int size);



#endif
