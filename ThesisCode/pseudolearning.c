#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "general.h"
#include "net.h"
#include "training.h"
#include "learning.h"
#include "sampling.h"
#include "pseudolearning.h"

/*
add a pattern pat at the end of the pseudo items
*/
int addPseudoItem(netinfo * net, traininginfo * training, pattern * pat){
	int i = training->numPseudoPatts; //i is the current total of Pseudo items

	if(i<training->maxPseudoPatts){
		copyPattern(net,&(training->pseudoPatts[i]),&(net->pat));
		sortPatternInput(net,&(training->pseudoPatts[i]));
		training->pseudoPatts[i].num = i;
		training->numPseudoPatts++;
		training->pseudoPatts[i].frequency = 1;
		training->pseudoPatts[i].cycleDist = -1;
		training->pseudoPatts[i].minimumDistToLearntPattern = -1;
		return i;
	}
	else{
		if (net->debugLevel & 1)fprintf(stderr,"Too many distinct pseudo items \n");
		return -1;
	}
}


/*
Test a new pseudo pattern against all the current pseudo items
*/
int compareToPseudoPatterns(netinfo * net, traininginfo * training, pattern * pat){
	int i,hammingDist,minDist=net->N;
	FOR(i,training->numPseudoPatts){//check against all pseudo patterns that have been stored
		hammingDist = equalState(net,pat,&(training->pseudoPatts[i]));
		if (hammingDist == 0){
			return i; //This is the position at which we found the pat in the pseudo items
		}
		minDist = MIN(hammingDist,minDist);
	}
	return NOTFOUND;
}





/*
This funtion is used to generate the pseudo items stored in the training->pseudoPatts
*/

int generatePseudoPatterns(netinfo * net, traininginfo * training){
	int i,dist,count,ERcount;
	int cycling=0;
	double cutoffLevel=training->pseudoRehearsalCutoff;
	int pattern, learntPattsinPseudoPop=0;
	count = 0;
	ERcount =0;
	training->numPseudoPatts = 0;
	if (training->useMaxLowestLearntvsCutoff){
		cutoffLevel = MAX(training->pseudoRehearsalCutoff,training->minLearntPatternRatio);
	}/*This is a little artificial as it uses a max of two variables.  
	 I can calc the lowest learnt ratio and if all of them are
	 High I will use the lowest of them otherwise I will use the ratio cutoff 
	 given in the param file
	 */
	do{
		setNetworkRandom(net);
		dist = relaxNetwork(net); //number of cycles to relax to the pattern
		sortPatternInput(net,&(net->pat));
		pattern = compareToLearntPatterns(net,training,&(net->pat));
		if ((training->pseudoRehearsalReal && (pattern != NOTFOUND)) || (net->pat.ratioStable > cutoffLevel)) {
			if (dist < net->maxCycles){  //If it is actually stable 
				if (training->uniquePseudoPatts){
					if (compareToPseudoPatterns(net,training,&(net->pat))==NOTFOUND){
						if (training->removeLearntFromPseudoPatts){
							if (pattern==NOTFOUND){
								addPseudoItem(net,training,&(net->pat));  // add if not found
							}
						} else {
							if (pattern!=NOTFOUND) learntPattsinPseudoPop++;
							addPseudoItem(net,training,&(net->pat));
						}

					}
				} else { // accept all patterns
					if (training->removeLearntFromPseudoPatts){
						if (compareToLearntPatterns(net,training,&(net->pat))==NOTFOUND){
							if (pattern!=NOTFOUND) learntPattsinPseudoPop++;
							addPseudoItem(net,training,&(net->pat));
						}
					} else {
						if (pattern!=NOTFOUND) learntPattsinPseudoPop++;
						addPseudoItem(net,training,&(net->pat));
					}
				}
			} else {
				cycling++;
			}
		} else {
			ERcount++;
		}
		count++;
	}while((training->numPseudoPatts<training->maxPseudoPatts)&& (count < training->maxSamplesForPseudoPatts));
	printf("%d %d %d learnt num = %d ",cycling, count, ERcount, learntPattsinPseudoPop);
	return count;
}


double deltaLearnPseudorehearsal(netinfo * net, traininginfo * training,int start, int end){
	int i,patNum;
	double totError,cycError,currError;
	int epoch;
	int totalEpochs;
	int * randArray=NULL;
	double learningConst;
	int * randArrayLearnt=NULL;

	if (start == 0) {// we are in the initial training block
		totalEpochs=training->initialTrainingEpochs;
	} else {
		totalEpochs=training->trainingEpochs;
	}

	training->learningConstDecay=training->learningConst;
	totError =0;
	printf("Pseudo Rehearsal Delta: %d\n",training->numPseudoPatts);
	epoch = 0;
	randArray = randomizeOrder(randArray,end - start); /*allocates an array with numbers 0 - (end-start)-1  ie
													   an array starting from 0 with end-start numbers in it*/
	randArrayLearnt = randomizeOrder(randArrayLearnt,start); /*allocates an array with numbers 0 - (end-start)-1  ie
															 an array starting from 0 with end-start numbers in it*/
	do{
		if(net->learnTiming == ONLINE){
			randomizeArray(randArray,end - start);
		}
		cycError = 0;
		totError = totError*training->errorTail;
		resetWeightDeltas(net);
		for(i=start;i<end;i++){
			if(net->learnTiming == ONLINE){
				patNum = start + randArray[i- start]; /* the array randArray contains values 0 - ((end-start)-1).  This is
													  considered the offset for the actual pattern numbers */
			}else{
				patNum = i;
			}
			if (patNum < training->maxLearntPatts){
//				calcPattInputs(net,training,&(training->learntPatts[patNum])); No longer needed as done in next step
				cycError += updateWeightDeltasNoise(net,training,&(training->learntPatts[training->learntOrder[patNum]]),FALSE);

				if(net->learnTiming == ONLINE){
					updateNetWeights(net,training,&(training->learntPatts[training->learntOrder[patNum]]));
				}
				if(net->debugLevel & 1) printf("<%d>",training->learntOrder[patNum]);
			}
		}
		if(net->learnTiming == BATCH){
            updateNetWeights(net,training,&(training->learntPatts[training->learntOrder[patNum]]));
        }        

		/* real rehearsal */
		if (training->realRehearsal){
			randomizeArray(randArrayLearnt,start);
			for(i=0;i<(start*training->realRehearsalPercent);i++){
				patNum = randArrayLearnt[i];
				calcPattInputs(net,training,&(training->learntPatts[training->learntOrder[patNum]]));
				if (training->noiseOnPseudoPatts){
					currError = updateWeightDeltasNoise(net,training,&(training->learntPatts[training->learntOrder[patNum]]),TRUE);
				} else {
					currError = updateWeightDeltasNoNoise(net,training,&(training->learntPatts[training->learntOrder[patNum]]),TRUE);
				}

				if (training->includePseudoInError){
					cycError+=currError;
				}
				if(net->learnTiming == ONLINE){
					updateNetWeights(net,training,&(training->learntPatts[training->learntOrder[patNum]]));
				}
				if(net->debugLevel & 1) printf("<p%d>",i);           
			}
			if(net->learnTiming == BATCH){
				updateNetWeights(net,training,&(training->learntPatts[training->learntOrder[patNum]]));
			}

		}else{
			/* Now do the pseudo rehearsal
			*/  

			for(i=0;i<training->numPseudoPatts;i++){
				calcPattInputs(net,training,&(training->pseudoPatts[i]));
				if (training->noiseOnPseudoPatts){
					currError = updateWeightDeltasNoise(net,training,&(training->pseudoPatts[i]),TRUE);
				} else {
					currError = updateWeightDeltasNoNoise(net,training,&(training->pseudoPatts[i]),TRUE);
				}

				if (training->includePseudoInError){
					cycError+=currError;
				}
				if(net->learnTiming == ONLINE){
					updateNetWeights(net,training,&(training->pseudoPatts[i]));
				}
				if(net->debugLevel & 1) printf("<p%d>",i);           
			} 
			if(net->learnTiming == BATCH){
				updateNetWeights(net,training,&(training->pseudoPatts[i]));
			}
		}
		/*Add lots of stuff here for pseudo rehearsal*/




		epoch++;
		totError += cycError;
		if(training->allEpochLearning) totError = 1;

		net->temperature -= (net->startTemperature / totalEpochs);
		training->learningConstDecay -= (training->learningConst-training->learningConstDecayTo)/totalEpochs;
		//		printf(": %10.3f\t: %8.3f\t:avg weight %6.3f  epoch:%4d learningConst %6.4f\r",totError,cycError,avgWeight(net),epoch,training->learningConstDecay);

	}while((epoch < totalEpochs)&&(totError > training->errorCriteria));
	training->minLearntPatternRatio=(training->minLearntPatternRatio*0.5)+(0.5*training->pseudoRehearsalCutoff); //This makes sure that the top end decays toward the cutoff level
	for(i=start;i<end;i++){
		sortPatternInput(net,&(training->learntPatts[training->learntOrder[i]]));
		training->minLearntPatternRatio=MIN(training->minLearntPatternRatio,training->learntPatts[training->learntOrder[i]].ratioStable);
		/*      printf("%2d %2d %2d: %10.3f\t: %8.3f\t:avg weight %6.3f  epoch:%4d min ratio : %6.4lf\n",
		start,end, training->numPseudoPatts,totError,cycError,avgWeight(net),epoch,training->minLearntPatternRatio
		);/*This should calculate the lowest learnt pattern ratio and keep a running total over many 
		patterns : issue how do you know what the lowest learnt pattern is currently in terms of ratio
		Need some mechanism that allows us to lower ratio realistically and not include too many
		spurious pattern.
		Could just use a lowest of the learnt and then cale by 0.8 for each learning epoch.
		That would give an interesting decay curve, and I could have the
		cut of level as the asymtopic value of the learning

		*/
	}
	printf("%2d %2d %2d: %10.3f\t: %8.3f\t:avg weight %6.3f  epoch:%4d min ratio : %6.4lf\n",
		start,end, training->numPseudoPatts,totError,cycError,avgWeight(net),epoch,training->minLearntPatternRatio
		);
	printf("\n");
	free(randArray);
	free(randArrayLearnt);
	return totError;
}



