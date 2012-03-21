#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "general.h"
#include "net.h"
#include "training.h"
#include "display.h"
#include "sampling.h"
/********************************************************************************************************

*/
void printNetwork(netinfo * net){
	int i;
	int columns = net->N/2;
	FOR(i,net->N){
		if (net->OFF==net->pat.unit[i]) 
			printf(".");
		else
			printf("#");
		if(i%(columns)==(columns-1)) printf("\n");
	}
	if(i%(columns)!=0) printf("\n");
}


/********************************************************************************************************

*/
void printNetInputSummary(netinfo * net){
	/* Note:  If you are using noise on net input then these will look different to non noisy
	versions printed elsewhere*/

	printf("min: %5.1f mean: % 7.2f  most: %5.1f -- Stdev: %6.3f Rel: % 5.3f Ratio: % 5.3f Guess: % 5.3f ",
		net->pat.leastStable,
		net->pat.meanStable,
		net->pat.mostStable,
		net->pat.stddev,
		net->pat.coefficientOfVariation,
		net->pat.ratioStable,
		(net->pat.meanStable -(net->pat.stddev*1.8) ) / (net->pat.meanStable + (net->pat.stddev*1.8))
		);
	if(net->calcHammingDistRelax){
		printf(" HamDist: %2d ",net->pat.hammingDist);
	}
}


/********************************************************************************************************

*/
void printNetInput(netinfo * net){
	int i;
	if (net->gnuplotGraphs){
		printf("\n");
		FOR(i,net->N){
			printf("%d %5.2f\n",i,(net->pat.unitNetInput[i]));
		}
	}
	else
	{
		FOR(i,net->N){
			printf("%2d ",(int)(net->pat.unitNetInput[i]));
		}
	}
	printNetInputSummary(net);
	printf("\n");
}




void printWeights(netinfo * net){
	int to,from;    
	printf("\n");
	FOR(to,net->N){
		FOR(from,net->N+(net->bias)){
			printf(" % 5.1f",(net->weight[to][from]));
		}
		printf("\n");
	}
}

/*
This function prints out the weights in integer bundels.  The array is from -10*N to +10N
this is mapped to the array 0 - 20*N.  So the zero position is 10*net->N.
*/
void printWeightsSummary(netinfo * net){
	int to,from,top,bottom,l,tot;
	int * numOfWeights;
	int zeroPos = (net->N*10);

	tot =0;
	numOfWeights = (int *)calloc(net->N*20,sizeof(int));
	printf("\n");
	FOR(to,net->N){
		FOR(from,net->N+(net->bias)){
			numOfWeights[zeroPos+(int)((int)floor(net->weight[to][from]+0.49))]++;
			tot++;
		}
	}
	bottom=1;
	top=(net->N*20)-1;
	while((numOfWeights[bottom] == 0)&& (numOfWeights[top] == 0)){
		bottom++;
		top--;
	}
	if(net->gnuplotGraphs){
		printf("# Val : Num \n");    
		for(l=bottom;l<=top;l++){
			printf("% 4d %4d\n",l-zeroPos,numOfWeights[l]);
		};
	}else{
		printf("Num: ");    
		for(l=bottom;l<=top;l++){
			printf(" %4d",numOfWeights[l]);
		}
		printf("\nVal: ");    
		for(l=bottom;l<=top;l++){
			printf(" % 4d",l-zeroPos);
		}
	}
	printf("\n");
	free(numOfWeights);
}


/********************************************************************************************************

*/
void displayPatt(netinfo * net, pattern * pat,int * order){
	int i;
	int columns = MIN(sqrt(net->N),100);
	if (order == NULL){
		FOR(i,net->N){
			if (net->OFF==pat->unit[i]) 
				printf(".");
			else
				printf("#");
			if(i%(columns)==(columns-1)) printf("\n");
		}
		if(i%(columns)!=0) printf("\n");
	}else{
		FOR(i,net->N){
			if (net->OFF==pat->unit[order[i]]) 
				printf(".");
			else
				printf("#");
			if(i%(columns)==(columns-1)) printf("\n");
		}
		if(i%(columns)!=0) printf("\n");
	}
}



void displayPattInput(netinfo * net,pattern * pat,int * order){
	int i;

	if(net->gnuplotGraphs){
		printf("Unit Input: \n"); 
		if (order == NULL){
			FOR(i,net->N){
				printf("%d %5.3f\n",i,(pat->unitInput[i]));
			}        
		}else{
			FOR(i,net->N){
				printf("%d %5.3f\n",i,(pat->unitInput[order[i]]));
			}
		}
		printf("End Unit Input:\n");
	}else{
		printf("Unit Input: "); 
		if (order == NULL){
			FOR(i,net->N){
				printf(" %5.3f",(pat->unitInput[i]));
			}        
		}else{
			FOR(i,net->N){
				printf(" %5.3f",(pat->unitInput[order[i]]));
			}
		}
		printf("\n");
	}
}


void displayPattNetInput(netinfo * net,pattern * pat){
	int i;

	if (net->gnuplotGraphs){
		printf("Net Input: \n");
		FOR(i,net->N){
			printf("%d %5.3f\n",i,(pat->unitNetInput[i]));
		}
		printf("End Net Input\n");
	}else{
		printf("Net  Input: ");
		FOR(i,net->N){
			printf(" %5.3f",(pat->unitNetInput[i]));
		}
		printf("\n");
	}

}

void displayPattSummary(netinfo * net, traininginfo * training, pattern * pat, int learning){
	double basinSize;
	int i; 
	double negative=-10000;
	double positive= 10000;
	double val;
	FOR(i,net->N)
	{
		val = pat->unitInput[i];
		if (val < 0) {
			negative = MAX(negative,val);
		}else{
			positive = MIN(positive,val);
		}
	}


	printf("Pat %3d:> min: % 6.2f  mean: % 7.2f  most: %6.2f -- Stdev %7.3f Rel: % 6.3f Ratio: % 5.3f Guess % 5.3f  Freq: %4d  CycAvg: %7.3f ",
		pat->num,
		pat->leastStable,
		pat->meanStable,
		pat->mostStable,
		pat->stddev,
		pat->coefficientOfVariation,
		pat->ratioStable,
		(pat->meanStable -(pat->stddev*1.8) ) / (pat->meanStable + (pat->stddev*1.8)),
		pat->frequency,
		pat->cycleDist/(double)pat->frequency);
	printf("- NearP %3d HamDist: %3d",
		pat->closestLearntPattern,
		pat->minimumDistToLearntPattern);
	if (net->calcInitialChanges > 0){
		printf(" InitCh: % 4.2f ", pat->initialChanges/(double)pat->frequency);
	}
	if ((pat->leastStable > -0.0000001)&&
			((training->calcSpuriousBasins && learning==spurious)||
			 (training->calcLearntBasins && learning==learnt))
		){
		basinSize = sampleBasinOfAttraction(net,training,pat);
		printf(" BasinSize: %7.5lf", basinSize);

	}
	printf(" Sat: %6.3lf",positive-negative);	
	printf("\n");
}

void displayPatternResults(netinfo * net, traininginfo * training,int * order){
	int i;

	printf(" patterns \n");
	FOR(i,training->maxLearntPatts){
		if (stablePattern(net,&(training->learntPatts[training->learntOrder[i]]))){
			if (training->displayLearntPattUnits)     {displayPatt(net,&(training->learntPatts[training->learntOrder[i]]),order);}        
			if (training->displayLearntPattInput)     {displayPattInput(net,&(training->learntPatts[training->learntOrder[i]]),order);}
			if (training->displayLearntPattNetInput)  {displayPattNetInput(net,&(training->learntPatts[training->learntOrder[i]]));}
			if (training->displayLearntPattSummary)   {printf("Learnt   ");displayPattSummary(net,training,&(training->learntPatts[training->learntOrder[i]]),learnt);}
		}
	}

	if (training->displaySpuriousPatts){
		printf("---------------\n");
		FOR(i,training->numSpuriousPatts){
			if (training->displaySpuriousPattUnits)    {displayPatt(net,&(training->spuriousPatts[i]),order);}
			if (training->displaySpuriousPattInput)    {displayPattInput(net,&(training->spuriousPatts[i]),order);}
			if (training->displaySpuriousPattNetInput) {displayPattNetInput(net,&(training->spuriousPatts[i]));}
			if (training->displaySpuriousPattSummary)  {printf("Spurious ");displayPattSummary(net,training,&(training->spuriousPatts[i]),spurious);}
		}
	}
}




void displayRatioResultsSummary(netinfo * net, traininginfo * training, singleRun * multiTrialSummary){
	singleRun a[1];

	initRun(a);
	calcValuesRun(net,training,a);
	displayValuesRun(net,training,a);
	addValuesSummary(multiTrialSummary,a);



}

void initRun(singleRun * a){
	a->maxFequencySpurious=0;
	a->minFequencyLearnt=10000;
	a->maxRatioSpurious=0;
	a->minRatioLearnt=1;
	a->minBasinSpurious=1;
	a->maxBasinLearnt=0;
	a->maxStdVarLearnt=0;
	a->minStdVarSpurious = 10000;
	a->maxMeanSpurious=0;
	a->minMeanLearnt=10000;
	a->maxLowSpurious=0;
	a->minLowLearnt =10000;
	a->maxCycleDist = 0;
	a->misClassifiedRatio = 0;
	a->misClassifiedLeastStable =0;
	a->misClassifiedRatioSet = 0;
	a->classifiedRatioSet = 0;
	a->misClassifiedStdVar = 0;
	a->misClassifiedStdVarSet = 0;
	a->classifiedStdVarSet = 0;
	a->misClassifiedMean = 0;
	a->misClassifiedNum= 0;
	a->RatioAStdVarDisagree =0;
	a->numLearntUnstable =0;
	a->numLearntStable =0;
	a->numLearntStableSqr =0;
	a->numBPStable =0;
	a->numBPStableSqr =0;
	a->numLearntFound =0;
	a->sumLearntFrequency=0;
	// long sumCircling=0; //again not used
	a->sumSpuriousFrequency=0;  
}


void calcValuesRun(netinfo * net, traininginfo * training, singleRun * a){
	int i;
	int numStable=0;
	int numBPStable=0;
	pattern pat;
	a->numLearntPatts = training->numLearntPatts;
	a->numSpuriousPatts= training->numSpuriousPatts;
	FOR(i,training->numLearntPatts){
		if (training->learntPatts[training->learntOrder[i]].stable == 1){ // Only check with stable patterns
			pat = training->learntPatts[training->learntOrder[i]];
			a->minFequencyLearnt=MIN(a->minFequencyLearnt,pat.frequency);
			a->minLowLearnt=MIN(a->minLowLearnt,pat.leastStable);
			a->sumLearntFrequency+=pat.frequency;
			a->minMeanLearnt=MIN(a->minMeanLearnt,pat.meanStable);
			a->minRatioLearnt=MIN(a->minRatioLearnt,pat.ratioStable);
			a->maxBasinLearnt=MAX(a->maxBasinLearnt,pat.basinSize);
			a->maxStdVarLearnt=MAX(a->maxStdVarLearnt,pat.coefficientOfVariation);
			a->maxCycleDist = MAX(a->maxCycleDist,pat.cycleDist);
			numStable++;
			if (i<training->initialNumberPatts) numBPStable++;
			if (XOR(pat.ratioStable > training->pseudoRehearsalCutoff,
				pat.coefficientOfVariation < training->pseudoRehearsalCutoff))
				a->RatioAStdVarDisagree++;            

			if (pat.frequency > 0) 
				a->numLearntFound++;
			if (pat.ratioStable > training->pseudoRehearsalCutoff)
				a->classifiedRatioSet++;
			if (pat.coefficientOfVariation < training->pseudoRehearsalCutoff)
				a->classifiedStdVarSet++;
		} else {
			a->numLearntUnstable++;
		}
	}
	a->numLearntStable+=numStable;
	a->numLearntStableSqr+=SQR(numStable);
	a->numBPStable+=numBPStable;
	a->numBPStableSqr+=SQR(numBPStable);


	FOR(i,training->numSpuriousPatts){
		if (XOR(training->spuriousPatts[i].ratioStable > training->pseudoRehearsalCutoff,
			training->spuriousPatts[i].coefficientOfVariation < training->pseudoRehearsalCutoff))
			a->RatioAStdVarDisagree++;
		if (training->spuriousPatts[i].ratioStable > a->minRatioLearnt )
			a->misClassifiedRatio++;
		if (training->spuriousPatts[i].ratioStable > training->pseudoRehearsalCutoff)
			a->misClassifiedRatioSet++;
		if (training->spuriousPatts[i].coefficientOfVariation < a->maxStdVarLearnt)
			a->misClassifiedStdVar++;
		if (training->spuriousPatts[i].coefficientOfVariation < training->pseudoRehearsalCutoff)
			a->misClassifiedStdVarSet++;
		if (training->spuriousPatts[i].meanStable > a->minMeanLearnt )
			a->misClassifiedMean++;
		if (training->spuriousPatts[i].frequency > a->minFequencyLearnt)
			a->misClassifiedNum++;
		if (training->spuriousPatts[i].leastStable > a->minLowLearnt)
			a->misClassifiedLeastStable++;

		a->sumSpuriousFrequency+=training->spuriousPatts[i].frequency;
		a->maxFequencySpurious=MAX(a->maxFequencySpurious,training->spuriousPatts[i].frequency);
		a->maxMeanSpurious=MAX(a->maxMeanSpurious,training->spuriousPatts[i].meanStable);
		a->maxRatioSpurious=MAX(a->maxRatioSpurious,training->spuriousPatts[i].ratioStable);
		a->minBasinSpurious=MIN(a->minBasinSpurious,training->spuriousPatts[i].basinSize);
		a->minStdVarSpurious =MIN(a->minStdVarSpurious,training->spuriousPatts[i].coefficientOfVariation);
		a->maxLowSpurious = MAX(a->maxLowSpurious,training->spuriousPatts[i].leastStable);
	}
}



void displayValuesRun(netinfo * net, traininginfo * training,singleRun *a){
	printf("\n");
	printf("          Freq:  Mean:   Ratio:  Basin:\nSpurious %5.0f %8.3f %7.4f %7.4f maximums independant\nLearnt   %5.0f %8.3f %7.4f %7.4f minimums independant\n",
		a->maxFequencySpurious,
		a->maxMeanSpurious,
		a->maxRatioSpurious,
		a->minBasinSpurious,
		a->minFequencyLearnt,
		a->minMeanLearnt,
		a->minRatioLearnt,
		a->maxBasinLearnt);
	printf("     MisClassified \n     Pats , Stb , Fnd , Spr :  -Rt  , %3.2f Rt (s,p) , -StdVar , %3.2f StdVar LeastSt: DisAgr , Mean ,  Num | %%Learn  %%Cycling \n",
		training->minLearntPatternRatio,
		training->pseudoRehearsalCutoff);
	printf("num:%5d %5d <%5d> %5d %5d : %5d  (%5d %5d) %7d   (%5d %5d) %5d : %6.3f  %5d  %5d | %5.2f %5.2f \n",
		a->numLearntPatts,
		a->numLearntStable,
		a->numBPStable,
		a->numLearntFound,
		a->numSpuriousPatts, //:
		a->misClassifiedRatio,
		a->misClassifiedRatioSet, //(
		a->classifiedRatioSet, //)
		a->misClassifiedStdVar,
		a->misClassifiedStdVarSet,//(
		a->classifiedStdVarSet,//)
		a->misClassifiedLeastStable, //:

		a->RatioAStdVarDisagree/(double)(a->numLearntStable+training->numSpuriousPatts),
		a->misClassifiedMean,
		a->misClassifiedNum,
		a->sumLearntFrequency*100.0/(double)training->numCheck,
		(training->numCheck -(a->sumSpuriousFrequency+a->sumLearntFrequency))*100.0/(double)training->numCheck);
	printf("             Ratio          StdVar          Least Stable    Mean \n");
	printf("vals : %5d (%5.3f > %5.3f) (%5.3f < %5.3f) (%5.0f > %5.0f) (%5.0f > %5.0f) \n",
		a->numLearntPatts,
		a->minRatioLearnt,
		a->maxRatioSpurious,
		a->maxStdVarLearnt,
		a->minStdVarSpurious,
		a->minLowLearnt,
		a->maxLowSpurious,
		a->minMeanLearnt,
		a->maxMeanSpurious
		);
}  

void displayFinalSummaryStabilityofPosition(netinfo * net, traininginfo * training,singleRun *a){
	int i; 
	double meanSquared;
	double meanOfSquares;
	int pats = (int)(a->numLearntPatts);
	FOR(i,pats){
		printf("%5.3f ",(a->patternsStabilityOverManyTrials[i]/(double)training->numTrials));
	}
	printf("\n");
}

void displayFinalSummaryDistanceTraveled(netinfo * net, traininginfo * training,singleRun *a){
	int i;
	int pats = (int)(a->numLearntPatts);
	FOR(i,pats){
		printf("%5.3f ",(a->patternsDistOverManyTrials[i]/(double)training->numTrials));
	}
	printf("\n");
}

void displayFinalSummaryStabilityofPositionBasin(netinfo * net, traininginfo * training,singleRun *a){
	int i;
	int pats = (int)(a->numLearntPatts);
	FOR(i,pats){
		if((double)a->patternsStabilityOverManyTrials[i]>0.000001){
			printf("%5.3f ",(a->patternsStabilityBasinOverManyTrials[i]/(double)a->patternsStabilityOverManyTrials[i]));
		}else{
			printf("%5.3f ",1.0);
		}
	}
	printf("\n");
}

void displayFinalSummaryStabilityofPositionBasinError(netinfo * net, traininginfo * training,singleRun *a){
	int i;
	int pats = (int)(a->numLearntPatts);
	FOR(i,pats){
		if((double)a->patternsStabilityOverManyTrials[i]>0.000001){
			printf(" %5.3f ",sqrt(
				(
				(a->patternsStabilityBasinSumSqrOverManyTrials[i] -
				(SQR(a->patternsStabilityBasinOverManyTrials[i])/(double)a->patternsStabilityOverManyTrials[i])
				)/
				(double)a->patternsStabilityOverManyTrials[i]
			)
				)
				);
		}else{
			printf("%5.3f ",0.0);
		}
	}
	printf("\n");
}

void displayFinalSummary(netinfo * net, traininginfo * training, singleRun *a){


	printf("num : %5d %7.2f %7.5f <%7.2f %7.5f > %7.2f %7.2f : %7.2f  (%7.2f %7.2f) %9.2f   (%7.2f %7.2f) %7.2f : %6.3f  %7.2f  %7.2f | %5.2f %5.2f ",
		a->numLearntPatts,
		a->numLearntStable/(double)training->numTrials,
		sqrt((a->numLearntStableSqr/(double)training->numTrials)-SQR(a->numLearntStable/(double)training->numTrials)),
/*<*/	a->numBPStable/(double)training->numTrials, 
		sqrt((a->numBPStableSqr/(double)training->numTrials)-SQR(a->numBPStable/(double)training->numTrials)),
		a->numLearntFound/(double)training->numTrials,
		a->numSpuriousPatts/(double)training->numTrials, 
/*:*/	a->misClassifiedRatio/(double)training->numTrials,
		a->misClassifiedRatioSet/(double)training->numTrials,
		a->classifiedRatioSet/(double)training->numTrials,
		a->misClassifiedStdVar/(double)training->numTrials,
		a->misClassifiedStdVarSet/(double)training->numTrials,
		a->classifiedStdVarSet/(double)training->numTrials,
		a->misClassifiedLeastStable/(double)training->numTrials, //:

		(a->RatioAStdVarDisagree/(double)(a->numLearntStable+a->numSpuriousPatts))/(double)training->numTrials,
		a->misClassifiedMean/(double)training->numTrials,
		a->misClassifiedNum/(double)training->numTrials, //|

		(a->sumLearntFrequency*100.0/(double)training->numCheck)/(double)training->numTrials,
		((training->numCheck -(a->sumSpuriousFrequency+a->sumLearntFrequency))*100.0/(double)training->numCheck)/(double)training->numTrials);
	printf("vals : %5d (%5.3f > %5.3f) (%5.3f < %5.3f) (%5.0f > %5.0f) (%5.0f > %5.0f)",
		a->numLearntPatts,
		a->minRatioLearnt/(double)training->numTrials,
		a->maxRatioSpurious/(double)training->numTrials,
		a->maxStdVarLearnt/(double)training->numTrials,
		a->minStdVarSpurious/(double)training->numTrials,
		a->minLowLearnt/(double)training->numTrials,
		a->maxLowSpurious/(double)training->numTrials,
		a->minMeanLearnt/(double)training->numTrials,
		a->maxMeanSpurious/(double)training->numTrials
		);
	printf(" compare: %7.2f %8.3f %7.4f %7.4f %7.1f %8.3f %7.4f %7.4f \n",
		a->maxFequencySpurious/(double)training->numTrials,
		a->maxMeanSpurious/(double)training->numTrials,
		a->maxRatioSpurious/(double)training->numTrials,
		a->minBasinSpurious/(double)training->numTrials,
		a->minFequencyLearnt/(double)training->numTrials,
		a->minMeanLearnt/(double)training->numTrials,
		a->minRatioLearnt/(double)training->numTrials,
		a->maxBasinLearnt/(double)training->numTrials);
}     

void addValuesSummary(singleRun * multiTrialSummary, singleRun * a){
	multiTrialSummary->numLearntPatts= a->numLearntPatts;
	multiTrialSummary->numSpuriousPatts+= a->numSpuriousPatts;
	multiTrialSummary->maxFequencySpurious+= a->maxFequencySpurious;
	multiTrialSummary->minFequencyLearnt+= a->minFequencyLearnt;
	multiTrialSummary->maxRatioSpurious+= a->maxRatioSpurious;
	multiTrialSummary->minRatioLearnt+= a->minRatioLearnt;
	multiTrialSummary->minBasinSpurious+= a->minBasinSpurious;
	multiTrialSummary->maxBasinLearnt+= a->maxBasinLearnt;
	multiTrialSummary->minStdVarSpurious += a->minStdVarSpurious;
	multiTrialSummary->maxMeanSpurious+= a->maxMeanSpurious;
	multiTrialSummary->minMeanLearnt+= a->minMeanLearnt;
	multiTrialSummary->maxLowSpurious+= a->maxLowSpurious;
	multiTrialSummary->minLowLearnt += a->minLowLearnt;
	multiTrialSummary->maxCycleDist += a->maxCycleDist;
	multiTrialSummary->misClassifiedRatio += a->misClassifiedRatio;
	multiTrialSummary->misClassifiedLeastStable += a->misClassifiedLeastStable;
	multiTrialSummary->misClassifiedRatioSet += a->misClassifiedRatioSet;
	multiTrialSummary->classifiedRatioSet += a->classifiedRatioSet;
	multiTrialSummary->misClassifiedStdVar += a->misClassifiedStdVar;
	multiTrialSummary->misClassifiedStdVarSet += a->misClassifiedStdVarSet;
	multiTrialSummary->classifiedStdVarSet += a->classifiedStdVarSet;
	multiTrialSummary->misClassifiedMean += a->misClassifiedMean;
	multiTrialSummary->misClassifiedNum+= a->misClassifiedNum;
	multiTrialSummary->RatioAStdVarDisagree += a->RatioAStdVarDisagree;
	multiTrialSummary->numLearntUnstable += a->numLearntUnstable;
	multiTrialSummary->numLearntStable += a->numLearntStable;
	multiTrialSummary->numLearntStableSqr += a->numLearntStableSqr;
	multiTrialSummary->numBPStable += a->numBPStable;
	multiTrialSummary->numBPStableSqr += a->numBPStableSqr;
	multiTrialSummary->numLearntFound += a->numLearntFound;
	multiTrialSummary->sumLearntFrequency+= a->sumLearntFrequency;
	// long sumCircling+= a->0; //again not used
	multiTrialSummary->sumSpuriousFrequency+= a->sumSpuriousFrequency;
}


void displayAvgStabProfile(netinfo * net, traininginfo * training){
	int pat,i,numLearntStable=0;

	double sum;
	double sumOfSquares;
	double num;
	double variance;

	double * avgStabProfileLearnt;
	double * avgStabProfileLearntStd;
	double * avgStabProfileLearntMin;
	double * avgStabProfileLearntMax;
	double * avgStabProfileSpurious;
	double * avgStabProfileSpuriousStd;
	double * avgStabProfileSpuriousMin;
	double * avgStabProfileSpuriousMax;


	avgStabProfileLearnt = calloc(net->N,sizeof(double));
	avgStabProfileLearntStd = calloc(net->N,sizeof(double));
	avgStabProfileLearntMin = calloc(net->N,sizeof(double));
	avgStabProfileLearntMax = calloc(net->N,sizeof(double));
	avgStabProfileSpurious = calloc(net->N,sizeof(double));
	avgStabProfileSpuriousStd = calloc(net->N,sizeof(double));
	avgStabProfileSpuriousMin = calloc(net->N,sizeof(double));
	avgStabProfileSpuriousMax = calloc(net->N,sizeof(double));

	FOR(i,net->N){ 
		avgStabProfileLearntMin[i] = 10000;
		avgStabProfileSpuriousMin[i] = 10000;};

		FOR(pat,training->maxLearntPatts){
			if (training->learntPatts[training->learntOrder[pat]].stable == 1){ // Only check with stable patterns
				numLearntStable++;
				FOR(i,net->N){
					avgStabProfileLearnt[i] += training->learntPatts[training->learntOrder[pat]].unitNetInput[i];
					avgStabProfileLearntStd[i] += SQUARE(training->learntPatts[training->learntOrder[pat]].unitNetInput[i]); //for working out std dev
					avgStabProfileLearntMin[i] = MIN(avgStabProfileLearntMin[i],training->learntPatts[training->learntOrder[pat]].unitNetInput[i]);
					avgStabProfileLearntMax[i] = MAX(avgStabProfileLearntMax[i],training->learntPatts[training->learntOrder[pat]].unitNetInput[i]);            }
			} 
		}
		FOR(i,net->N){
			sum = avgStabProfileLearnt[i];
			sumOfSquares = avgStabProfileLearntStd[i];
			num = (double)numLearntStable;
			variance = (sumOfSquares - (sqr(sum)/num))/(num-1);
			/*
			variance=   sum of squares	 -  (square the sum / number of items /number of items) / number of items
			Standard Deviation = square root of[ (sum of Xsquared -((sum of X)*(sum of X)/N))/ (N-1)) ]
			*/
			avgStabProfileLearntStd[i] = sqrt(variance);

			avgStabProfileLearnt[i] /= (double)numLearntStable;
			/*The sum of squares -*/
		}

		FOR(pat,training->numSpuriousPatts){
			if (training->spuriousPatts[pat].stable == 1){ // Only check with stable patterns
				FOR(i,net->N){
					avgStabProfileSpurious[i] += training->spuriousPatts[pat].unitNetInput[i];
					avgStabProfileSpuriousStd[i] += SQUARE(training->spuriousPatts[pat].unitNetInput[i]); //for std
					avgStabProfileSpuriousMin[i] = MIN(avgStabProfileSpuriousMin[i],training->spuriousPatts[pat].unitNetInput[i]);
					avgStabProfileSpuriousMax[i] = MAX(avgStabProfileSpuriousMax[i],training->spuriousPatts[pat].unitNetInput[i]);
				}
			} 
		}
		FOR(i,net->N){
			sum = avgStabProfileSpurious[i];
			sumOfSquares = avgStabProfileSpuriousStd[i];
			num = (double)training->numSpuriousPatts;
			variance = (sumOfSquares - (sqr(sum)/num))/(num-1);
			/* vari  =   sum of squares	 -  (square the sum / number of items /number of items) / number of items*/
			avgStabProfileSpuriousStd[i] = sqrt(variance);
			avgStabProfileSpurious[i] /= (double)training->numSpuriousPatts;
		}

		//Now comes the displaying part of this function
		printf("\nLearnt patterns avgStabProfile:\n");
		printf("num     avg       std dev     min        max\n");
		FOR(i,net->N){
			printf("%3d % 10.5f % 10.5f % 10.5f % 10.5f\n",
				i,avgStabProfileLearnt[i],avgStabProfileLearntStd[i],avgStabProfileLearntMin[i],avgStabProfileLearntMax[i]);
		}
		printf("Spurious patterns avgStabProfile:\n");
		printf("num     avg       std dev     min        max\n");
		FOR(i,net->N){
			printf("%3d % 10.5f % 10.5f % 10.5f % 10.5f\n"
				,i,avgStabProfileSpurious[i],avgStabProfileSpuriousStd[i], 				avgStabProfileSpuriousMin[i],avgStabProfileSpuriousMax[i]);
		}


		free(avgStabProfileLearnt);
		free(avgStabProfileLearntStd);
		free(avgStabProfileLearntMin);
		free(avgStabProfileLearntMax);
		free(avgStabProfileSpurious);
		free(avgStabProfileSpuriousStd);
		free(avgStabProfileSpuriousMin);
		free(avgStabProfileSpuriousMax);

}    

void saveStabilityProfilePattern(netinfo * net, traininginfo * training, FILE * stabilityProfileFile, int pat){
	int i;
	FOR(i,net->N){
		fprintf(stabilityProfileFile,"%8.4f\n",training->learntPatts[training->learntOrder[pat]].unitNetInput[i]);
	}
	fprintf(stabilityProfileFile,"\n\n");
}

void saveStabilityProfileSpuriousPattern(netinfo * net, traininginfo * training, FILE * stabilityProfileFile, int pat){
	int i;
	FOR(i,net->N){
		fprintf(stabilityProfileFile,"%8.4f\n",training->spuriousPatts[pat].unitNetInput[i]);
	}
	fprintf(stabilityProfileFile,"\n\n");
}

void saveSaturationProfilePattern(netinfo * net, traininginfo * training, FILE * saturationProfileFile, int pat){
	int i;
	FOR(i,net->N){
		fprintf(saturationProfileFile,"%8.4f\n",training->learntPatts[training->learntOrder[pat]].unitInput[i]);
	}
	fprintf(saturationProfileFile,"\n\n");
}

void saveSaturationProfileSpuriousPattern(netinfo * net, traininginfo * training, FILE * saturationProfileFile, int pat){
	int i;
	FOR(i,net->N){
		fprintf(saturationProfileFile,"%8.4f\n",training->spuriousPatts[pat].unitInput[i]);
	}
	fprintf(saturationProfileFile,"\n\n");
}

