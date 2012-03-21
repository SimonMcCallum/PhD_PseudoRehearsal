/*
	File:	net.c
	Author:	Simon McCallum
	Description:

*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "general.h"
#include "random.h"
#include "net.h"
#include "allocation.h"


extern void seedRandom(long seedValue);
extern long initializeRandomSeed();
extern int nextRandomInt();
extern double nextRandomDouble();


/********************************************************************************************************
Set all weights in net->weights to 0
*/
int initWeights(netinfo * net){
    int to,from;
    FOR(to,net->N){
        FOR(from,net->N+(net->bias)){
            net->weight[to][from]=0;
        }
    }    
    return to;
}



/*
This is the standard decay function
multiplies each weight by 1-net->weightDecayValue.
*/
int decayWeights(netinfo * net){
    int to,from;
    FOR(to,net->N){
        FOR(from,net->N+(net->bias)){
            net->weight[to][from]*=(double)(1-(double)(net->weightDecayValue));
        }
    }    
    return to;
}



/*
This function is here so that when you want to change a weight you need to check a few things.  This is the unified location to check the weight values restriction.  This function currently checks the capping value.  I have separated this so that more restrictions can be added later.
*/
double addToWeight(netinfo * net, double * weight, double value){
    double change;
    
    switch(net->weightCappingType){
    case NOCAP: 
        *(weight) += value;
        break;
    case HARDCAP:
        *(weight) += value;
        if (*(weight) > net->weightCappingValue)
            *(weight) = net->weightCappingValue;
        else if (*(weight) < -net->weightCappingValue)
            *(weight) =  -net->weightCappingValue;
        break;
    case SOFTCAP://Initially linear later I'll use a nice smooth function
        change =  (fabs(*(weight))-net->weightCappingValue)/(double)net->weightCappingValue;
        if (change < 0)
            *(weight) += value;  
        else if (change < 1)
            *(weight) += value * change;
        else
            //no change as capping at 2* net->weightCappingValue
            ;
        }
    return *(weight);
}


/*
This function normalises the connections to a unit so that they
sum to a set value.  Generally not a good plan to normalise to early this is held off by the fact that
the normalisation only runs if the unit is GREATER than a set value.

The other type of normalisation is to normalise across all the units.  This is the NORMALIZENET
option.  This makes the total weights across the entire network stay BELOW a set value.

Normalisation should only occur after a set number of patterns have been learn.  This is to
stop the network fixating on the first pattern/s learnt. */

double normaliseWeights(netinfo * net){
    int to,from;
    double sum,ratio;
    
    switch(net->weightNormalisationType){
    case NORMALISEUNIT:
        FOR(to,net->N){
            sum = 0;
            FOR(from,net->N+(net->bias)){
                sum += fabs(net->weight[to][from]);
            }
            ratio = net->weightNormalisationValue/sum;
            if (ratio<1){ 
                FOR(from,net->N+(net->bias)){
                    net->weight[to][from] *= ratio;
                }
            }            
        }
        break;
    case NORMALISEUNIT_ZEROMEAN:
        FOR(to,net->N){
            sum = 0;
            FOR(from,net->N+(net->bias)){
                sum += net->weight[to][from];
            }
            ratio = sum/net->N;
            if (ratio<1){ 
                FOR(from,net->N+(net->bias)){
                    net->weight[to][from] -= ratio;
                }
            }            
        }
        break;

	case NORMALISEUNIT_BOTH:
        FOR(to,net->N){
            sum = 0;
            FOR(from,net->N+(net->bias)){
                sum += fabs(net->weight[to][from]);
            }
            ratio = net->weightNormalisationValue/sum;
            if (ratio<1){ 
                FOR(from,net->N+(net->bias)){
                    net->weight[to][from] *= ratio;
                }
            }            
        }
		FOR(to,net->N){
            sum = 0;
            FOR(from,net->N+(net->bias)){
                sum += net->weight[to][from];
            }
            ratio = sum/net->N;
            if (ratio<1){ 
                FOR(from,net->N+(net->bias)){
                    net->weight[to][from] -= ratio;
                }
            }            
        }
        break;

    case NORMALISENET://Initially linear later I'll use a nice smooth function
        sum = 0;
        FOR(to,net->N){
            FOR(from,net->N+(net->bias)){
                sum += fabs(net->weight[to][from]);
            }
        }
        ratio = (net->weightNormalisationValue*net->N)/sum;
        if (ratio<1){ 
            FOR(to,net->N){
                FOR(from,net->N+(net->bias)){
                    net->weight[to][from] *= ratio;
                }
            }   
        }
    }
    return sum;
}



/*
set unitErrors to 0
*/
void resetErrors(netinfo *net){	
    int i;
    FOR(i,net->N){
        net->pat.unitError[i] =0;
    }   
}

/*
reset the nets weight deltas
*/
void resetWeightDeltas(netinfo *net){	
    int to,from;
    FOR(to,net->N){
        FOR(from,(net->N+(net->bias))){
            net->weightDelta[to][from] =0;
        }   
    }
}


/********************************************************************************************************
Return the activation of the unit using the heavy side funtion
*/
double activation(netinfo * net, double input){
 if (net->activationType == THRESHOLD) 
    return (input > 0) ? net->ON : net->OFF;   
 else
    return (input > 0) ? net->ON : net->OFF;
 
}

/********************************************************************************************************
Return the activation as a sigmoid function responce to the input value passed in 
*/
double activationSigmoid(netinfo * net, double input){
    if (net->OFF == 0)
        return (1 /(double)(1 + exp(-input)));
    else
        return (((1 /(double)(1 + exp(-input)))*2)-1);
}


/********************************************************************************************************
Set each of the units inputs to a random double between +net->sampleProbOn and -(1-net->sampleProbOn)
*/
int setNetworkRandom(netinfo * net){
    int i;
    FOR(i,net->N){
        net->pat.unitInput[i]=nextRandomDouble()-(1-net->sampleProbOn);
        net->pat.unit[i] = activation(net,net->pat.unitInput[i]);
    }
    net->pat.unit[net->N]=net->ON;
    return i;
}

/*selects an item and removes it from the array.  It also swaps
in the number from the end of the array and decreases the number
of item, count, in the array  This could be a little bit of a 
problem if we don't keep a close eye on the array */
int selectWithoutReplacement(int * order, int * count){
  int item,pos;
  pos = RandomInt(0,(*count)-1);
  item = order[pos];
  order[pos]=order[(*count)-1];
  order[(*count)-1]=item;
  (*count)= (*count)-1;
  return item;
}


int setNetworkPartial(netinfo * net, pattern * pat, double overlap){
  int i,j;
  int item;
  int count = net->N;
  int numToFlip = (int)((net->N)*(double)(1-overlap));

  /* Set up pattern and then disrupt by ovelap */
  FOR(i,net->N){
    net->pat.unit[i]=pat->unit[i];
  }
  
  /*set up order of units*/
  count = net->N;
  FOR(i,count){
    net->unitOrder[i]=i;
  }

  FOR(j,numToFlip){
    item = selectWithoutReplacement(net->unitOrder,&(count));
    net->pat.unit[item] = (net->pat.unit[item]==net->ON)?
      net->OFF:net->ON;
  }
  net->pat.unit[net->N]=net->ON;
  return numToFlip;
}


/*
Tests to see if two patterns have the same values. 
*/
int identicalState(netinfo * net, pattern * left, pattern * right){
    int i,hamDist=0;
    for(i=0;i<net->N;i++){
        if (!withinMargin(left->unit[i],right->unit[i])) 
            hamDist++; 				//Calc Hamming Distance
    }
    return hamDist;
}

/*The return of 0 for no differences is strange but makes it consistent with
The hamming distance version of this function and other C functions
that return 0 for no difference*/
int fastIdenticalState(netinfo * net, pattern * left, pattern * right){
    int i;
    for(i=0;i<net->N;i++){
        if (!withinMargin(left->unit[i],right->unit[i])) 
            return 1; 				//Stop as soon as they are different
    }
    return 0;
}


/*
Check to see if the state of two patterns are inverses of each other. 
*/
int inverseState(netinfo * net, pattern * left, pattern * right){
    int i,hamDist=0;
    for(i=0;i<net->N;i++){
        if (withinMargin(left->unit[i],right->unit[i])) 
            hamDist++; 				//Calc Hamming Distance
    }
    return hamDist;
}

int fastInverseState(netinfo * net, pattern * left, pattern * right){
    int i;
    for(i=0;i<net->N;i++){
        if (withinMargin(left->unit[i],right->unit[i])) 
            return 1; 				//Stop as soon as they are different
    }
    return 0;
}




/********************************************************************************************************
This does equal to both real unit and the inverse of the left unit
Returns Hamming distance, */

int equalState(netinfo * net, pattern * left, pattern * right){
    return MIN(identicalState(net,left,right),inverseState(net,left,right));
}

/*
  Carefully note that this function returns 0 if the patterns are identical
This is consistent with the hamming distance version of the code
interpret the 1 as there being a difference and 0 as identical
 */
int fastEqualState(netinfo * net, pattern * left, pattern * right){
  return (fastIdenticalState(net,left,right)&&fastInverseState(net,left,right));
}




double avgWeight(netinfo * net){
    int to,from;
    double sum,avg;
	if(net->calcWeightAverages){
		sum =0;
		FOR(to,net->N){
			FOR(from,net->N+(net->bias)){
				sum += fabs(net->weight[to][from]);
			}
		}
		avg = (double)sum/(double)(net->N*(net->N+(net->bias)));
	    return avg;
	} else {
		return 0.0;
	}
}

int calcInputs(netinfo * net){
    int to,from;
    double sum = 0;
    FOR(to,net->N){
        net->pat.unitInput[to] = 0;
        FOR(from,(net->N+(net->bias))){
			if(to != from)
	            net->pat.unitInput[to] += net->weight[to][from] * net->pat.unit[from];
        }
        sum += fabs(net->pat.unitInput[to]);
    }
    net->pat.meanUnitInput = sum/(double)net->N;
    return to;
}

int calcOutputsSynchronous(netinfo * net){
    int to;
    double old;
    int sum=0;
    FOR(to,net->N){
        old = net->pat.unit[to];
        net->pat.unit[to] = activation(net,net->pat.unitInput[to]);
        if (!withinMargin(old,net->pat.unit[to]))
            sum++;
    }
    return sum;
}

int calcOutputsAsynchronous(netinfo * net, double partialUpdate){
    int i,unit;
    double old;
    int sum=0;
    FOR(i,net->N*partialUpdate){
        unit = RandomInt(0,net->N);
        old = net->pat.unit[unit];
        net->pat.unit[unit] = activation(net,net->pat.unitInput[unit]);
        if (!withinMargin(old,net->pat.unit[unit]))
            sum++;
    }
    return sum;
}


/********************************************************************************************************

*/

int stable(netinfo * net){
  int i;
  double min=1000;
  FOR(i,net->N){
    if(activation(net,net->pat.unitInput[i]) != net->pat.unit[i])
      return 0;
  }
  return 1; // all the patterns input and output must be the same
}
 


int unstable(netinfo * net){
    int i;
    double min=1000;
    FOR(i,net->N){
        net->pat.unitNetInput[i] = (activation(net,net->pat.unitInput[i]) == net->pat.unit[i])? fabs(net->pat.unitInput[i]) : -fabs(net->pat.unitInput[i]);
        min = (min>net->pat.unitNetInput[i])?net->pat.unitNetInput[i]:min;
    }
    if (min<0) 
        return 1;
    else
        return 0;
}

int numUnstable(netinfo * net){
    int i;
    int count=0;
    FOR(i,net->N){
        net->pat.unitNetInput[i] = (activation(net,net->pat.unitInput[i]) == net->pat.unit[i])? fabs(net->pat.unitInput[i]) : -fabs(net->pat.unitInput[i]);
        if(net->pat.unitNetInput[i]<0){count++;}
    }
	return count;
}

/*
This function applies a Gaussian random adjustment to the input of each unit.
*/
void blurNetInputs(netinfo *net, double * input){
    int i;
    if (net->gaussianNoiseType == RELATIVE){
        FOR(i,net->N){
            input[i] = net->pat.unitInput[i] + RandomGaussian(0,net->pat.meanUnitInput*net->gaussianRelativeRangeForRelaxation);
        }
    } else {
        FOR(i,net->N){
            input[i] = net->pat.unitInput[i] + RandomGaussian(0,net->gaussianAbsoluteRangeForRelaxation);
        }
    }
}


/* This is the biggy.  This is the code to relax the network to a stable unit */

int relaxNetworkSynchronous(netinfo * net){
    int cycle, changes;
    int stableTime=0;
    int initChanges=0;
	int to,from;
	double old;
    
    cycle = 0;
    if (net->calcHammingDistRelax)
      setState(net,&(net->initPat),&(net->pat));        
    do{
        calcInputs(net);
		/*
        if (net->noiseDuringRelaxation)
            blurNetInputs(net,net->pat.unitInput);
        changes = calcOutputsSynchronous(net);
		*/
		changes = 0;
		FOR(to,net->N){
			net->pat.unitInput[to] = 0;
			FOR(from,(net->N+(net->bias))){
				net->pat.unitInput[to] += net->weight[to][from] * net->pat.unit[from];
			}
			old=net->pat.unit[to];
			net->pat.unit[to]=activation(net,net->pat.unitInput[to]); //permuted update
			if (net->pat.unit[to] != old) changes++;
			//if(net->pat.unit[to] != activation(net,net->pat.unitInput[to])) changes++
		}
        if(changes == 0){
            stableTime++;
        } else {
            stableTime = 0;
			if ((cycle==0) && (net->calcInitialChanges)){
				initChanges = changes;
			} 
//		  if(cycle>0) printf("cycle %d %d \n",cycle,changes);
		  cycle++;
        } 
			

    }while((cycle<net->maxCycles)&&(stableTime < net->lenghtOfTimeStableInRelaxation));
    if (cycle < net->maxCycles){
      if(unstable(net))fprintf(stderr,"Bugger relaxation is broken sync\n");
    }
	if(net->calcHammingDistRelax){
      net->pat.hammingDist = equalState(net,&(net->pat),&(net->initPat));
	}
	if(net->calcInitialChanges){
      net->pat.initialChanges = initChanges;
	  //this is where I need to link with the pattern number
	}

    return cycle;
}

/*
This calculates the number of units that need to change returns this as the int and puts the numbers
into the unitsToChange array.
*/
int findNumToChange(netinfo * net, int * unitsToChange){
    int i;
    int sum = 0;
    
    FOR(i,net->N){
        if (!(withinMargin(net->pat.unit[i],activation(net,net->pat.unitInput[i]))))
            unitsToChange[sum++] = i;
    }
    return sum;
}

/*
This calculates the number of units that need to change with NOISE returns this as the int and puts the numbers
into the unitsToChange array.
*/
int findNumToChangeNoise(netinfo * net, int * unitsToChange, double * noisyInput){
    int i;
    int sum = 0;
    
    blurNetInputs(net,noisyInput);
    FOR(i,net->N){
        if (!withinMargin(net->pat.unit[i],activation(net,net->pat.unitInput[i])))
            unitsToChange[sum++] = i;
    }
    return sum;
}

double calcHopfieldNetworkEnergy(netinfo * net){
    int i;
    double sum = 0;
    
    FOR(i,net->N){
        if (!withinMargin(net->pat.unit[i],activation(net,net->pat.unitInput[i])))
            sum += fabs(net->pat.unitInput[i]);
    }
    return sum;
}

double calcAllNetworkEnergy(netinfo * net){
    int i;
    double sum = 0;
    
    FOR(i,net->N){
        if (!withinMargin(net->pat.unit[i],activation(net,net->pat.unitInput[i])))
            sum += fabs(net->pat.unitInput[i]);
        else
            sum -= fabs(net->pat.unitInput[i]);
    }
    return sum;
}


/*
This does the task of updating the input to the units in the network
*/
int updateInputsUnitChanged(netinfo * net, int from, double oldActivation){
    int to;
    double newActivation;
    int sum = 0;
    double change = 0;
    double sumInput =0;
    
    /*step through all the inputs and change them by the weight of the connection
    from changed unit "unit" to the unit i */
    newActivation = net->pat.unit[from];
    FOR(to,net->N){
        change = -(net->weight[to][from]*oldActivation) +
                (net->weight[to][from]*newActivation);
        net->pat.unitInput[to] += change;
        sumInput += fabs(net->pat.unitInput[to]);
        sum += (int)change;
    }
    net->pat.meanUnitInput = sumInput/(double)net->N;
    return sum;    
}

/*
*/
int relaxNetworkAsynchronous(netinfo * net){
    int cycle;
    int unit;
    double oldActivation;
    int stableTime=0;    
    int numToChange=0;

 
    if (net->calcHammingDistRelax)
      setState(net,&(net->initPat),&(net->pat));        
    
    cycle = 0;
    /* This is the faster version of relaxation.  The idea is that we
    record the Input to the units and when a unit changes we update 
    the input to the units by their connection to the unit that has
    changed.  This changes an N^2 algorithm to an N algorithm.
    
    Also instead of a random choose of any unit we make a random
    choose from units that are wanting to change.  Ie we have a list
    of the units that want to change and we select from that list the
    unit that will change.*/
    calcInputs(net); //Global calculation

    if (net->noiseDuringRelaxation){
        do{
            /* alg plan
            having calculated inputs assign the units that want to change O(N); This could be done
                in the unit input into the unitToChange array O(N);
            select one of these at random        
            undate it        
            undate the unit inputs to reflect the change in this unit 
            */
            numToChange = findNumToChangeNoise(net,net->unitsToChange,net->noisyInput);
            unit = net->unitsToChange[RandomInt(0,numToChange)];
            oldActivation = net->pat.unit[unit];
            net->pat.unit[unit]=activation(net,net->noisyInput[unit]);
            updateInputsUnitChanged(net,unit,oldActivation);
            if (numToChange == 0)
                stableTime++;
            else {
                stableTime = 0;
                cycle++;
            }
        }while((cycle<net->maxCycles)&&(stableTime < net->lenghtOfTimeStableInRelaxation));
    } else { //Noiseless relaxation
      numToChange = findNumToChange(net,net->unitsToChange);
      while((cycle<net->maxCycles)&&(numToChange > 0)){
		unit = net->unitsToChange[RandomInt(0,numToChange-1)];
		oldActivation = net->pat.unit[unit];
		net->pat.unit[unit]=activation(net,net->pat.unitInput[unit]);
		updateInputsUnitChanged(net,unit,oldActivation);
		//        calcInputs(net); //This removed for speed replaced by above line
		cycle++;
		numToChange = findNumToChange(net,net->unitsToChange);

		//            energy = calcAllNetworkEnergy(net);
      }
    }
    
    if (cycle < net->maxCycles)
        if(unstable(net))fprintf(stderr,"Oh dear relaxation is broken async\n");
    if(net->calcHammingDistRelax)
      net->pat.hammingDist = equalState(net,&(net->pat),&(net->initPat));
    return cycle;
}


/*
 For this stability check what we need to do is go through each unit and find out it's input
 and then pass this through the activation function and check that the result is the same
 as the current activation.
 1->N calc input and compaor to current value
 return: 0 if UNSTABLE, 1 if stable
 */
int stablePattern(netinfo *net, pattern *pat){
  int to,from;
  double unitInput = 0;
  for(to=0;to<net->N;to++){
    unitInput = 0;
    for(from=0;from<(net->N+(net->bias));from++){
      unitInput += net->weight[to][from] * pat->unit[from];
    }
    if (activation(net,unitInput)!=pat->unit[to]) return 0;
  }
  return 1;
  
}

int relaxNetwork(netinfo * net){
    if (net->relaxation == SYNCHRONOUS){
        return relaxNetworkSynchronous(net);
    } else {
        return relaxNetworkAsynchronous(net);
    }
}





int numcmp(void const * a, void const * b){
    if(withinMargin(*(double *)a,*(double *)b)) return 0;
    if(*(double *)a<*(double *)b) return -1;
    return 1;
}
/*******************************************************************

This function calculates the ratio of the most stable units in relation to the least stable units.
*/
double sortPatternInput(netinfo * net, pattern * pat){
    
    int i;
    double low=0,high=0,sum=0;
    double variance;
    double sumsquares=0;

    FOR(i,net->N){
        pat->unitNetInput[i]=(activation(net,pat->unitInput[i]) == pat->unit[i])? fabs(pat->unitInput[i]) : -fabs(pat->unitInput[i]);
        sum += pat->unitNetInput[i];
        sumsquares += SQUARE(pat->unitNetInput[i]);
    }
    qsort(pat->unitNetInput, net->N, sizeof(double),
             (numcmp));
    
    FOR(i,(int)(net->N*net->unitsInRatio)){
        low +=pat->unitNetInput[i];
        high +=pat->unitNetInput[net->N-(i+1)];
    } 
  
    variance = (sumsquares - (SQUARE(sum)/(double)net->N))/(double)(net->N-1);
    pat->leastStable = pat->unitNetInput[0];
    pat->meanStable  = sum/(double)net->N;
    pat->mostStable  = pat->unitNetInput[net->N-1];
    pat->stddev = sqrt(variance);
    pat->coefficientOfVariation = pat->stddev/(double)pat->meanStable;
    //    pat->stable = pat->leastStable>=0;
    pat->stable = pat->leastStable>=0;
    pat->ratioStable = low/high;
     
    return pat->unitNetInput[net->N-1];
    
}



/********************************************************************************************************

This function calculates the ratio of the most stable units in relation to the least stable units.
*/
double calcRatio2(netinfo * net){
    int i;
    double stability;
    double * lowEnd;
    double * highEnd;
    double highLowEnd = -1;
//    int highLowEndPos = -1;
//    double lowHighEnd = 100000000;
//    int lowHighEndPos = -1;
    
    lowEnd = malloc(sizeof(double)*(int)(net->N*net->unitsInRatio));
    highEnd = malloc(sizeof(double)*(int)(net->N*net->unitsInRatio));
    
    FOR(i,net->N){
        stability = net->pat.unit[i]*net->pat.unitInput[i]; //Old watch this space 
        if(stability < highLowEnd){
        
        }
    }
return 1.0;
    
}






/********************************************************************************************************

*/
int setState(netinfo * net, pattern * to, pattern * from){
    int i;
    FOR(i,net->N){
        to->unit[i]=from->unit[i];
    }
    from->unit[net->N]=net->ON;
    return i;
}


int copyPattern(netinfo * net,pattern * toPat, pattern * fromPat){
    int i;
    FOR(i,net->N+1){
        toPat->unit[i] = fromPat->unit[i];
        toPat->unitInput[i] = fromPat->unitInput[i];
        toPat->unitNetInput[i] = fromPat->unitNetInput[i];
        toPat->unitDelta[i] = fromPat->unitDelta[i];
        toPat->unitError[i] = fromPat->unitError[i];
    }
    toPat->frequency = fromPat->frequency;
    toPat->sum = fromPat->sum; //of units on
    toPat->cycleDist = fromPat->cycleDist;
    toPat->cycle = fromPat->cycle;
    toPat->stable = fromPat->stable;
    toPat->leastStable = fromPat->leastStable;
    toPat->meanStable = fromPat->meanStable;
    toPat->mostStable = fromPat->mostStable;
    toPat->stddev = fromPat->stddev;
    toPat->coefficientOfVariation = fromPat->coefficientOfVariation;
    toPat->ratioStable = fromPat->ratioStable;    
    toPat->meanUnitInput = fromPat->meanUnitInput;

    toPat->frequency = fromPat->frequency;
    toPat->minimumDistToLearntPattern = fromPat->minimumDistToLearntPattern;
    toPat->closestLearntPattern = fromPat->closestLearntPattern;
    toPat->initialChanges = fromPat->initialChanges;
    
    return 1;
    
}







