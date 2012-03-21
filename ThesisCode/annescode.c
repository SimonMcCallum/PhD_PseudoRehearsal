#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "general.h"
#include "random.h"
#include "net.h"
#include "training.h"
#include "allocation.h"
#include "display.h"
#include "sampling.h"
#include "learning.h"
#include "pseudolearning.h"
#include "unlearning.h"
#include "param.h"
#include "patternFileIO.h"


int main(int argc, char * argv[])
{


	int i,trial;

	int run;
	int stable;
	int currentBottomPat,currentTopPat;
	int dist;  
	netinfo * net;
	traininginfo * training;
	char *netParamFilename,*trainingParamFilename, *defaultNetParam, *defaultTrainingParam;
	int * order;
	long currentSeed;
	char * junk;
	char * baseFilename;
	double basinSize;

	FILE * stabilityProfileFile;
	char *stabilityProfileFilename;

	long summaryCount;
	singleRun * multiTrialSummaryUnlearning;
	singleRun * multiTrialSummary;
	int totalSummaries;
	long numLearntPatternsStable;



	baseFilename = (char *)calloc(1000, sizeof(char));
	stabilityProfileFilename= (char *)calloc(1000, sizeof(char));

	netParamFilename = (char *)calloc(1000, sizeof(char));
	trainingParamFilename = (char *)calloc(1000, sizeof(char));    
	defaultNetParam = (char *)calloc(1000,sizeof(char));
	defaultTrainingParam = (char *)calloc(1000, sizeof(char));    

	for(i=strlen(argv[0])-1;((i>0) && ((argv[0][i]!='/')&&(argv[0][i]!=92)));i--){
		/*printf("%c:%d:%d\n",argv[0][i],argv[0][i],i);
		empty*/};

	strncpy(baseFilename,argv[0],i+1);
	/*    printf("%s %d\n",baseFilename,i+1);*/
	strcat(baseFilename,"default");
	/*     printf("%s %d\n",baseFilename,i+1);*/
	strncpy(defaultNetParam,baseFilename,999); /*the .net will be added later*/
	strncpy(defaultTrainingParam,baseFilename,999);


	switch (argc) {
		case 0: case 1:
			if (strncmp("-h",argv[1],2)==0){
				fprintf(stderr,"Usage: %s <paramfile> | \n        %s <paramfile.net> <paramfile2.train>",argv[0],argv[0]); 
			}
			break;
		case 2:
			strncpy(baseFilename,argv[1],999);
			strncpy(netParamFilename,baseFilename,999);
			strncpy(trainingParamFilename,baseFilename,999);
			break;
		case 3:
			printf("%s %s\n",argv[1],argv[2]);
			sprintf(netParamFilename,"%s",argv[1]);    
			sprintf(trainingParamFilename,"%s",argv[2]);
			printf("%s %s\n",netParamFilename,trainingParamFilename);  
			break;
		case 4:
			freopen(argv[3], "w", stdout );
			printf("%s %s\n",argv[1],argv[2]);
			sprintf(netParamFilename,"%s",argv[1]);    
			sprintf(trainingParamFilename,"%s",argv[2]);
			printf("%s %s\n",netParamFilename,trainingParamFilename);  
			break;
		default: 
			strcpy(baseFilename,"default");
			strncpy(netParamFilename,baseFilename,999);
			strncpy(trainingParamFilename,baseFilename,999);
	}

	/*the default name of the parameter file is "default.net" */

	net = (netinfo *)calloc(1,sizeof(netinfo));
	training = (traininginfo *)calloc(1,sizeof(traininginfo));

	initNetParam(net);
	initTrainingParam(training);

	readNetParamFile(defaultNetParam);
	readTrainingParamFile(defaultTrainingParam);

	readNetParamFile(netParamFilename);
	readTrainingParamFile(trainingParamFilename);



	processNetParameters(net);
	processTrainingParameters(training);

	allocNetMemory(net);
	allocPatternMemory(net,training);

	if(paramDefined("randomSeed")){ 
		currentSeed = net->randomSeed;
	} else {
		currentSeed = initializeRandomSeed();
		addParamFileLong("randomSeed",currentSeed,"net");

	}
	
	seedRandom(currentSeed); //this makes it possible to rerun results
	/*To make the comparison of techiniques more uniform I
	seed the random patterns with the same seed this allows 
	me to directly compare the different learning parameters
	without running many trials to try to get a more global pattern
	When running multiple trials to a single learning mechanism
	this seed can be set from the command line.
	*/

	printParamDatabase();

	saveNetParamDatabase("out");
	saveTrainParamDatabase("out");

	order = (int *)calloc(net->N,sizeof(int));




	totalSummaries = MAX(0,((training->maxLearntPatts - training->initialNumberPatts)/training->step)+1);
	multiTrialSummaryUnlearning=(singleRun *)calloc(totalSummaries,sizeof(singleRun));
	multiTrialSummary=(singleRun *)calloc(totalSummaries,sizeof(singleRun));
	FOR(i,totalSummaries){
		multiTrialSummary[i].patternsStabilityOverManyTrials=(double *)calloc(training->maxLearntPatts,sizeof(double));
		multiTrialSummary[i].patternsStabilityOverManyTrialsSqr=(double *)calloc(training->maxLearntPatts,sizeof(double));
		multiTrialSummary[i].patternsDistOverManyTrials=(double *)calloc(training->maxLearntPatts,sizeof(double));
		multiTrialSummary[i].patternsStabilityBasinOverManyTrials=(double *)calloc(training->maxLearntPatts,sizeof(double));
		multiTrialSummary[i].patternsStabilityBasinSumSqrOverManyTrials=(double *)calloc(training->maxLearntPatts,sizeof(double));
	}


	for(trial=0;trial<training->numTrials;trial++){
		printf("\n\n**********************TRIAL Number %3d ********************************\n\n\n",trial);
		summaryCount=0;

		initAll(net,training);
		initPatternMemory(net,training);
		initWeights(net);

		training->numSpuriousPatts=0;
		training->numPseudoPatts=0;
		training->numUnlearningPatts=0;

		training->minLearntPatternRatio=1;
		//    training->unlearningConst=((trial+1)*0.0005);
		//    training->initialNumberPatts=trial;


		for(i=0;i<net->N;order[i]=i++);

		/*This bit of code checks to see where the patterns are comming from
		*/
		initSequence(training->learntOrder,training->maxLearntPatts);
		if(training->patternsFromFile){
			loadAllLearntPatterns(net,training);
			knuthShuffle(training->learntOrder,training->maxLearntPatts);
		} else {
			generateUnitInputs(net,training);
			generatePatterns(net,training);
		}


		//  printf("Hit return\n");
		//  getchar();
		//  initializeRandomSeed();
		/*this generates the random testing for the relaxation 
		sampling, needs to be different every time the network 
		is run*/
		run = 0;
		currentBottomPat = 0;
		currentTopPat = training->initialNumberPatts;
		while((currentTopPat <=  training->maxLearntPatts)){
			training->numSpuriousPatts = 0;
			training->numLearntPatts=currentTopPat;
			if(HEBBIAN == training->learningType){
				printf("Hebbian: **********************************************\n");
				hebbianLearnPartial(net, training, training->learntPatts, training->learntOrder, currentBottomPat,
					currentTopPat, training->learningConst);
			}else{
				if (STRICTDELTA == training->learningType){
					deltaLearnPartial(net,training,currentBottomPat,currentTopPat);
				}else{
					if (PSEUDODELTA == training->learningType){
						if (currentBottomPat >= training->initialNumberPatts) /*This is added so that there are some initial patterns that can be learnt befor the
																			  network tries to generate pseudo patterns*/
																			  generatePseudoPatterns(net,training);
						deltaLearnPseudorehearsal(net,training,currentBottomPat,currentTopPat);
					} else {
						if (UNLEARNING == training->learningType){
							printf("Hebbian: **********************************************\n");
							hebbianLearnPartial(net, training, training->learntPatts, training->learntOrder, currentBottomPat,
								currentTopPat, training->learningConst);
							/* The following has been removed so that unlearning will work on the inital population
							if (currentBottomPat >= training->initialNumberPatts) *//*This is added so that there are some initial patterns that can be learnt befor the
							network tries to generate unlearning patterns*/
							{
								//generateUnlearningPatterns(net,training);
								printf("Testing the Learnt Patterns BEFORE UNLEARNING\n");
								FOR(i,MIN(currentTopPat,training->maxLearntPatts)){
									//printf("\n");
									stable = setNetwork(net,training,&(training->learntPatts[training->learntOrder[i]]));
									//printNetwork(net);

									sortPatternInput(net,&(net->pat));
									printf("Pat %2d : stable = %2d\t",training->learntOrder[i],stable);
									dist = relaxNetwork(net);
									printNetInputSummary(net);
									printf("Dist = %d",dist);
									if((dist == 0)&&(training->calcLearntBasins)){
										basinSize = sampleBasinOfAttraction(net,training,&(training->learntPatts[training->learntOrder[i]]));
										printf(" BasinSize: %7.5lf",basinSize);
									}
									printf("\n");



								}
								sampleNet(net,training);
								if (training->reorderUnitsinPatts){
									printf("swaps = %d\n",sortAcrossMultiplePatterns(net,training,order));
								}
								displayPatternResults(net,training,order);
								displayRatioResultsSummary(net,training,&(multiTrialSummaryUnlearning[summaryCount]));

								printf("Unlearning: **********************************************\n");

								unlearnCycles(net,training,currentTopPat-currentBottomPat);
							}

						}
					}
				}
			}
			printf("Training Complete\n");
			printf("Testing the Learnt Patterns\n");
			numLearntPatternsStable = 0;
			FOR(i,MIN(currentTopPat,training->maxLearntPatts)){
				stable = setNetwork(net,training,&(training->learntPatts[training->learntOrder[i]]));
				//stable = 0 or 1

				//printNetwork(net);
				//dist = relaxNetwork(net);
				if (stable){
					multiTrialSummary[summaryCount].patternsStabilityOverManyTrials[i]+=1;
				}
				sortPatternInput(net,&(net->pat));
				dist = relaxNetwork(net);
				multiTrialSummary[summaryCount].patternsDistOverManyTrials[i]+=dist;
				if (training->displayTestedPatterns){
					printf("Pat %2d : stable = %2d\t",training->learntOrder[i],stable);
					printNetInputSummary(net);
					printf("Dist = %d",dist);
				}

				if((dist == 0)&&(training->calcLearntBasins)){
					basinSize = sampleBasinOfAttraction(net,training,&(training->learntPatts[training->learntOrder[i]]));
//					basinSize = sampleBasinOfAttraction(net,training,&(training->learntPatts[training->learntOrder[i]]));//repeatability
					if (training->displayTestedPatterns){
						printf(" BasinSize: %7.5lf",basinSize);
					}
					multiTrialSummary[summaryCount].patternsStabilityBasinOverManyTrials[i]+=basinSize;
					multiTrialSummary[summaryCount].patternsStabilityBasinSumSqrOverManyTrials[i]+=SQR(basinSize);
					/* don't need this unless I want the basin to look really small for patterns with errors
					} else {
					multiTrialSummary[summaryCount].patternsStabilityBasinOverManyTrials[i]+=1.0;
					multiTrialSummary[summaryCount].patternsStabilityBasinSumSqrOverManyTrials[i]+=SQR(1.0);
					*/
				}

				if (training->displayTestedPatterns)
					printf("\n");
			}

			if(net->debugLevel & 2) printWeights(net);  
			if(net->debugLevel & 1) printWeightsSummary(net);
			sampleNet(net,training);
			if (training->reorderUnitsinPatts){
				printf("swaps = %d\n",sortAcrossMultiplePatterns(net,training,order));
			}
			displayPatternResults(net,training,order);
			displayRatioResultsSummary(net,training,&(multiTrialSummary[summaryCount]));
			summaryCount++;
			if (net->gnuplotGraphs){
				displayAvgStabProfile(net,training);
			}


			if ((trial == 0)&&(training->saveStabilityProfilePatternCheck==TRUE)){
				FOR(i,training->maxLearntPatts){
					if(training->saveStabilityProfilePattern[i]==TRUE){
						sprintf(stabilityProfileFilename,"energyProfilePat%d.dat",i);
						if (run == 0) {
							stabilityProfileFile = fopen(stabilityProfileFilename,"w");
							fprintf(stabilityProfileFile,"#%s %s %d\n",argv[1],argv[2], currentSeed);
						}else{
							stabilityProfileFile = fopen(stabilityProfileFilename,"a");
						}
						fprintf(stabilityProfileFile,"#Top pat = %d\n",currentTopPat);
						saveStabilityProfilePattern(net,training,stabilityProfileFile,i);					
						fclose(stabilityProfileFile);

						qsort(training->learntPatts[training->learntOrder[i]].unitInput, net->N, sizeof(double),
						 (numcmp));

						sprintf(stabilityProfileFilename,"satProfilePat%d.dat",i);
						if (run == 0) {
							stabilityProfileFile = fopen(stabilityProfileFilename,"w");
							fprintf(stabilityProfileFile,"#%s %s %d\n",argv[1],argv[2], currentSeed);
						}else{
							stabilityProfileFile = fopen(stabilityProfileFilename,"a");
						}
						fprintf(stabilityProfileFile,"#Top pat = %d\n",currentTopPat);
						saveSaturationProfilePattern(net,training,stabilityProfileFile,i);					
						fclose(stabilityProfileFile);

					}
				}
			}
			if ((trial == 0)&&(training->saveStabilityProfilePatternCheck==TRUE)){
				FOR(i,training->numSpuriousPatsProfiles){
					if (run == 0) {
						sprintf(stabilityProfileFilename,"energyProfileSpurPat%d.dat",i);
						stabilityProfileFile = fopen(stabilityProfileFilename,"w");
							fprintf(stabilityProfileFile,"#%s %s %d\n",argv[1],argv[2], currentSeed);
						fclose(stabilityProfileFile);
						sprintf(stabilityProfileFilename,"satProfileSpurPat%d.dat",i);
						stabilityProfileFile = fopen(stabilityProfileFilename,"w");				
							fprintf(stabilityProfileFile,"#%s %s %d\n",argv[1],argv[2], currentSeed);
						fclose(stabilityProfileFile);
					}
					sprintf(stabilityProfileFilename,"energyProfileSpurPat%d.dat",i);
					stabilityProfileFile = fopen(stabilityProfileFilename,"a");
					fprintf(stabilityProfileFile,"#Top pat = %d\n",currentTopPat);
					if (training->numSpuriousPatts>i) {
						saveStabilityProfileSpuriousPattern(net,training,stabilityProfileFile,i);
					} else {
						fprintf(stabilityProfileFile,"\n\n");
					}
					fclose(stabilityProfileFile);

					qsort(training->spuriousPatts[i].unitInput, net->N, sizeof(double),
						(numcmp));

					sprintf(stabilityProfileFilename,"satProfileSpurPat%d.dat",i);
					stabilityProfileFile = fopen(stabilityProfileFilename,"a");				
					fprintf(stabilityProfileFile,"#Top pat = %d\n",currentTopPat);
					if (training->numSpuriousPatts>i) {
						saveSaturationProfileSpuriousPattern(net,training,stabilityProfileFile,i);					
					} else {
						fprintf(stabilityProfileFile,"\n\n");
					}
					fclose(stabilityProfileFile);

				}
			}

			currentBottomPat = currentTopPat;
			currentTopPat+=training->step;
			run+=1;
		}

	}
	/*only do this if we suspect an error*///printWeights(net);
	FOR(i,totalSummaries){
		if (UNLEARNING == training->learningType)
			displayFinalSummary(net,training,&(multiTrialSummaryUnlearning[i]));       
		displayFinalSummary(net,training,&(multiTrialSummary[i]));
	}

	FOR(i,totalSummaries){
		printf("stabilityAvg: ");
		displayFinalSummaryStabilityofPosition(net,training,&(multiTrialSummary[i]));
		printf("DistAvg: ");
		displayFinalSummaryDistanceTraveled(net,training,&(multiTrialSummary[i]));
		printf("basinsizeAvg: ");
		displayFinalSummaryStabilityofPositionBasin(net,training,&(multiTrialSummary[i]));
		printf("basinsizeErr: ");
		displayFinalSummaryStabilityofPositionBasinError(net,training,&(multiTrialSummary[i]));
	}

	printf("\n");

	return 0;
}




/* 2002/4/18 */
