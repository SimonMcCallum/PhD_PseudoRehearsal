#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "random.h"
#include "general.h"
#include "net.h"
#include "training.h"
#include "allocation.h"
#include "display.h"
#include "learning.h"



/********************************************************************************************************
Okay interesting addition to the hebbian learning alogrithm.  The addition concept is that highly stimulated
units can have some of there inputs corrupted (by noise) and still be stable in a learnt pattern.

This could be assumed to be similar to the idea that the more active you are the noiser your training is.  
Units near their decision surface need to accurately update.  Units further away can affort to sacrifice
some of their stability to other units.

Two possible mechanisms.  High input = NO learning
/                         High input = Noisy learning

noisy works well what about no learning  
*/
int hebbianLearnPartial(netinfo * net, traininginfo * training, pattern * patternArray, int * patternOrder, int start,int end, double learningConst){
    int pat,to,from,cycles;
    double probNoise;
    double probAdjustment;
    double noiseEffect;
    double probability;
    pattern * newPat; 
    net->bias = 0;
    FOR(cycles,training->hebbianCycles){
        for(pat=start;pat<end;pat++){
          newPat = &(patternArray[patternOrder[pat]]);
                     
            calcPattInputs(net,training,newPat);
            /*set the input for the learnt pattern (pat)  */
            FOR(to,net->N){
              if (!((training->learningType == UNLEARNING)
                    && (training->unlearningSuperHeated)
                    && (learningConst<0)
                    && (newPat->unitInput[to] < training->unlearningCutoffHigh))){
                /* this is the check for Unlearning the spuerheated states only */
                
                probNoise = fabs(newPat->unitInput[to]/(double)(net->N));
                noiseEffect = (RandomDouble(0,1)>=(training->hebbianNoiseLevel*probNoise));
                FOR(from,net->N+(net->bias)){
                    /*The prob adjustment is actually to calculate how common a change is
                    if prob on is 0.2 then the likely hood of an OFF OFF connection is 0.8*0.8 = 0.64
                    Thus there will be a lot of these.  To cope with this we adjust the weight
                    change to reflect the probability of connection  This keeps the weight 
                    valuesnicely distributed around 0.*/
                    if (training->adjustLearningForPobability){
                        probability = newPat->sum/(double)net->N;
                        probAdjustment = ((newPat->unit[to] == net->ON)?
                                        ((newPat->unit[from] == net->ON)?
                                            (1-probability)*(1-probability)://ON, ON
                                            -(1-probability)*(probability)) // ON, OFF
                                        :
                                        ((newPat->unit[from] == net->ON)?
                                            -(probability)*(1-probability):// OFF, ON
                                            (probability)*(probability)) //  OFF, OFF
                                      );

                        addToWeight(net,net->weight[to]+from, probAdjustment*noiseEffect*learningConst //(probAdjustment*((probability*(1-probability))/(double)(net->N-1)))*
                            ); //This is a binary value determining if the change will happen
                        
                    }else{
                        
                      // This is the raw learning system
                        addToWeight(net,net->weight[to]+from, 
                            ((newPat->unit[to] == newPat->unit[from]) ? 
                                (1 * learningConst) : (-1 * learningConst))*
                            noiseEffect); //This is a binary value determining if the change will happen
                            
                    }
                    
                }
                                /*This is a line that I have to justify later*/
  //              if (net->bias) addToWeight(net,net->weight[to]+N,+1 * learningConst);

                
                net->weight[to][to]=0;
              }
              else {
              }
            }
            if(net->symmetricWeights){
                createSymmetry(net);
            }
			if((learningConst>0)&& (net->weightDecay)){
                decayWeights(net);
            }
            if(net->weightNormalisationType != NONORMALISATION){
                normaliseWeights(net);
            }
            
            
            
        }
    }
    return pat;
}


int hebbianLearnAll(netinfo * net, traininginfo * training){
    return hebbianLearnPartial(net,training, training->learntPatts, training->learntOrder, 0, training->maxLearntPatts, training->learningConst);
}



int calcPattInputs(netinfo *net,traininginfo * training, pattern * setPat){
    int to,from;
    double sum=0;
    FOR(to,net->N){
        setPat->unitInput[to] = 0;
        FOR(from,net->N+(net->bias)){
            setPat->unitInput[to] += net->weight[to][from] * setPat->unit[from];
        }
        sum += fabs(setPat->unitInput[to]);
    }
    setPat->meanUnitInput = sum/(double)net->N;
    return to;
}


/*
This function applies a Gaussian random adjustment to the input of each unit.
*/

void blurPattInputs(netinfo *net,traininginfo * training, pattern * setPat, double ratio){
    int i;
    if (training->gaussianNoiseType== RELATIVE) {
        FOR(i,net->N){
            setPat->unitInput[i] += RandomGaussian(0,(setPat->meanUnitInput* training->gaussianRelativeRange)*ratio);
        }
    } else {
        FOR(i,net->N){
            setPat->unitInput[i] += RandomGaussian(0,training->gaussianAbsoluteRange*ratio);
        }
    }
}

/*This function just sets the vector of units in output to the activation of 
the input in the vector output.unitinput
*/
int calcPattOutputs(netinfo *net,traininginfo * training, pattern * output){
int i;
    if (DELTAERROR == training->errorCalcType){ 
        FOR(i,net->N){
            output->unit[i] = activation(net,output->unitInput[i]);
        }
    } else if (SIGMOIDERROR == training->errorCalcType) {
        FOR(i,net->N){
            output->unit[i] = activationSigmoid(net,output->unitInput[i]);
        }
    }
    return 1; //
}

double calcPattErrors(netinfo *net,traininginfo * training,pattern * desired, pattern *actual){	
    int i;
    double error = 0, theError = 0;
    error = 0;
	net->averageInput = net->N;
       FOR(i,net->N){
            theError = (double)(desired->unit[i] - actual->unit[i]);
            error += fabs(theError);
			// This is the addition to decrease high values
			if (theError == 0)
			{
				if(fabs(net->pat.unitInput[i]) > net->N*0.7) net->averageInput = (net->averageInput*0.9) + (fabs(net->pat.unitInput[i])/10.0);
				if(fabs(net->pat.unitInput[i]) > net->averageInput*1.5)
				{
					theError = -SG(net->pat.unitInput[i])*(training->learningConst/5);
					
				}

			}
			
//				 	  		d           -          f
            net->pat.unitError[i] = theError;
        }
    return (error/(double)net->N);    
}

/*This is the funtion that applies the hetro associative noise to the pattern NosiyPatt
ie noise just to the from pattern not the to patter*/
void applyHetroNoise(netinfo * net, traininginfo * training, pattern * noisyPat, double noiseLevel){
    int i;
    FOR(i,net->N){
        if(RandomUniform() < noiseLevel){
            noisyPat->unit[i] = (noisyPat->unit[i]==net->OFF)?net->ON:net->OFF;
        }
    }

}



/*This updates the deltas by learning const * error * act*/
int updateWeightDeltas(netinfo *net,traininginfo * training,pattern * inputPat, double ratio){
    int to,from;
    FOR(to,net->N){
        FOR(from,(net->N+(net->bias))){
			net->weightDelta[to][from] += training->learningConstDecay * net->pat.unitError[to] * inputPat->unit[from] * ratio;
        }
    }
    return to;
}

int updateWeightThermalDeltas(netinfo *net,traininginfo * training,pattern * inputPat, double thermal, double ratio){
    int to,from;
    FOR(to,net->N){
        FOR(from,(net->N+(net->bias))){
            net->weightDelta[to][from] += thermal * training->learningConstDecay * net->pat.unitError[to] * inputPat->unit[from] * ratio;
        }
    }
    return to;
}



/*Main goal:  adjust input for noise and then update the weight deltas
returns the error of the current pattern squared*/
double updateWeightDeltasNoise(netinfo * net, traininginfo * training, pattern * learningPat, BOOL PseudoPatt){
	double thermalValue;
    double error=0;
    pattern * pat = &(net->pat);
     
    if (training->noiseHetroAssociative){
    /*The step in applying noise are :
            allocate memory
            create a copy of the learnt pattern
            make it noisy
            set up the network for and iteration*/
        copyPattern(net,&(net->noisyPat),learningPat);
		if (PseudoPatt){
			applyHetroNoise(net,training,&(net->noisyPat),(training->hetroNoiseLevel*training->noiseRatioPseudoPatts));
		}else{
			applyHetroNoise(net,training,&(net->noisyPat),training->hetroNoiseLevel);
		}
        copyPattern(net,pat,&(net->noisyPat));
    } else {
        copyPattern(net,pat,learningPat); //copy the pattern to learn into the network
    }
    calcPattInputs(net,training,pat);
    if (net->debugLevel & 4) displayPattInput(net,pat,NULL);
    /*checking noise levels*/

	if (training->noiseOnInput){
		if (PseudoPatt){
			blurPattInputs(net,training,pat,training->noiseRatioPseudoPatts);
		}else{
			blurPattInputs(net,training,pat,1.0);
		}
       
	}
    if (net->debugLevel & 4) displayPattInput(net,pat,NULL);
    /*checking noise levels*/
    
    calcPattOutputs(net,training,pat);
    
    error = (double)calcPattErrors(net,training,learningPat,pat);
	thermalValue = (pow(E, -(fabs(error) / net->temperature)));
	if (PseudoPatt){
		if (training->noiseHetroAssociative){
			if (net->thermalDelta){ 
				updateWeightThermalDeltas(net,training,learningPat,thermalValue,training->noiseRatioPseudoPatts); //&(net->noisyPat),thermalValue);/*this needs to be that pattern that the network was initialised with*/
			} else {
				updateWeightDeltas(net,training,learningPat,training->noiseRatioPseudoPatts);//&(net->noisyPat));/*this needs to be that pattern that the network was initialised with*/
			}
		} else {
			if (net->thermalDelta){ 
				updateWeightThermalDeltas(net,training,learningPat,thermalValue,training->noiseRatioPseudoPatts);/*this needs to be that pattern that the network was initialised with*/
			} else {
				updateWeightDeltas(net,training,learningPat,training->noiseRatioPseudoPatts);
			}
		}
	}else{
		if (training->noiseHetroAssociative){
			if (net->thermalDelta){ 
				updateWeightThermalDeltas(net,training,learningPat,thermalValue,1.0); //&(net->noisyPat),thermalValue);/*this needs to be that pattern that the network was initialised with*/
			} else {
				updateWeightDeltas(net,training,learningPat,1.0);//&(net->noisyPat));/*this needs to be that pattern that the network was initialised with*/
			}
		} else {
			if (net->thermalDelta){ 
				updateWeightThermalDeltas(net,training,learningPat,thermalValue,1.0);/*this needs to be that pattern that the network was initialised with*/
			} else {
				updateWeightDeltas(net,training,learningPat,1.0);
			}
		}
	}
    return error*error; 
}


double updateWeightDeltasNoNoise(netinfo * net, traininginfo * training, pattern * learningPat, BOOL PseudoPatt){
	double thermalValue;
    double error=0;
    pattern * pat = &(net->pat);
     
    copyPattern(net,pat,learningPat); //copy the pattern to learn into the network
 
    calcPattInputs(net,training,pat);
    
    calcPattOutputs(net,training,pat);
    
    error = (double)calcPattErrors(net,training,learningPat,pat);

	thermalValue = (pow(E, -(fabs(error) / net->temperature)));
	if (PseudoPatt){
		if (net->thermalDelta){ 
			updateWeightThermalDeltas(net,training,learningPat,thermalValue,training->learningRatioPseudoPatts);/*this needs to be that pattern that the network was initialised with*/
		} else {
			updateWeightDeltas(net,training,learningPat,training->learningRatioPseudoPatts);
		}
	} else {
		if (net->thermalDelta){ 
			updateWeightThermalDeltas(net,training,learningPat,thermalValue,1.0);/*this needs to be that pattern that the network was initialised with*/
		} else {
			updateWeightDeltas(net,training,learningPat,1.0);
		}
	}
    return error*error; 
}




int updateNetWeights(netinfo *net,traininginfo * training,pattern * setPat){
    int to,from;
    FOR(to,net->N){
        FOR(from,(net->N+(net->bias))){
            addToWeight(net,net->weight[to]+from,net->weightDelta[to][from]);
        }
        net->weight[to][to]=0;
    }
    resetWeightDeltas(net);
    
    if(net->symmetricWeights){
        createSymmetry(net);
    }
    
    if(net->weightDecay){
        decayWeights(net);
    }
    if(net->weightNormalisationType != NONORMALISATION){
        normaliseWeights(net);
    }
    
    return to;
}


int createSymmetry(netinfo *net){
    int to,from;
    FOR(to,net->N){
        for(from=0;from<(net->N);from++){//no need for bias as bias does not need to be symmetric
            if (from < (net->N)-to){
                net->weight[to][from] = (net->weight[to][from]+net->weight[from][to])/2;
            }else{
                net->weight[to][from] = net->weight[from][to];
            }
        }
    }
    return to;
}



double deltaLearnAll(netinfo * net, traininginfo * training){
    return deltaLearnPartial(net,training, 0, training->maxLearntPatts);
}

double deltaLearnPartial(netinfo * net, traininginfo * training,int start, int end){
    int i,pat;
    double totError,cycError;
    int epoch;
    int totalEpochs;
    int * randArray=NULL;
	
	if (start == 0) {// we are in the initial training block
		totalEpochs=training->initialTrainingEpochs;
	} else {
		totalEpochs=training->trainingEpochs;
	}

	training->learningConstDecay=training->learningConst;
    totError =0;
    printf("Delta: \n");
    epoch = 0;
    randArray = randomizeOrder(randArray,end - start); /*allocates an array with numbers 0 - (end-start)-1  ie
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
                pat = start + randArray[i- start]; /* the array randArray contains values 0 - ((end-start)-1).  This is
                considered the offset for the actual pattern numbers */
            }else{
                pat = i;
            }
            calcPattInputs(net,training,&(training->learntPatts[training->learntOrder[pat]]));
 //           displayPatt(net,&(training->learntPatts[training->learntOrder[pat]]),NULL);
            cycError += updateWeightDeltasNoise(net,training,&(training->learntPatts[training->learntOrder[pat]]),FALSE);
            if(net->learnTiming == ONLINE){
                updateNetWeights(net,training,&(training->learntPatts[training->learntOrder[pat]]));
            }
	    //            printf("<%d>",pat);
        }
        if(net->learnTiming == BATCH){
            updateNetWeights(net,training,&(training->learntPatts[training->learntOrder[pat]]));
        }        
        epoch++;
        totError += cycError;
		if(training->allEpochLearning) totError = 1;  
        if(net->debugLevel & 1) printf(": %10.3f\t: %8.3f\t:avg weight %6.3f  epoch:%4d\n",totError,cycError,avgWeight(net),epoch);
    }while((epoch < totalEpochs)&&(totError > training->errorCriteria));
     printf(": %10.3f\t: %8.3f\t:avg weight %6.3f  epoch:%4d\n",totError,cycError,avgWeight(net),epoch);
     printf("\n");
    free(randArray);
    return totError;
}

/*
 Test a new pattern against all learnt patterns
 returns the position of the pattern in the list of patterns to learn
 or NOTFOUND (-1) if not present
 */
int compareToLearntPatterns(netinfo * net, traininginfo * training, pattern * pat){
    int i,hammingDist;
	pat->minimumDistToLearntPattern=net->N;
    FOR(i,training->numLearntPatts){//check against all learnt patterns
        hammingDist = equalState(net,pat,&(training->learntPatts[training->learntOrder[i]]));
		if(hammingDist<pat->minimumDistToLearntPattern){
			pat->minimumDistToLearntPattern = hammingDist;
			pat->closestLearntPattern = training->learntOrder[i];
		}
        if (hammingDist == 0){
			
            return training->learntOrder[i]; //This is the position at which we found the pat in the learnt items
        }
    }
    return NOTFOUND;
}




