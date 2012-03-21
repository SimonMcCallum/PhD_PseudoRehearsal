#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "general.h"
#include "net.h"
#include "training.h"
#include "learning.h"
#include "sampling.h"
#include "unlearning.h"

/*
add a pattern pat at the end of the unlearning items
*/
int addUnlearningItem(netinfo * net, traininginfo * training, pattern * pat){
    int i = training->numUnlearningPatts; //i is the current total of Unlearning items
    
    if(i<training->maxUnlearningPatts){
        copyPattern(net,&(training->unlearningPatts[i]),&(net->pat));
        sortPatternInput(net,&(training->unlearningPatts[i]));
        training->unlearningPatts[i].num = i;
        training->numUnlearningPatts++;
        training->unlearningPatts[i].frequency = 1;
        training->unlearningPatts[i].cycleDist = -1;
        training->unlearningPatts[i].minimumDistToLearntPattern = -1;
       return i;
        }
    else{
        if (net->debugLevel & 1)fprintf(stderr,"Too many distinct unlearning items \n");
        return -1;
        }
}


/*
Test a new unlearning pattern against all the current unlearning items
*/
int compareToUnlearningPatterns(netinfo * net, traininginfo * training, pattern * pat){
    int i,hammingDist,minDist=net->N;
    FOR(i,training->numUnlearningPatts){//check against all unlearning patterns that have been stored
        hammingDist = equalState(net,pat,&(training->unlearningPatts[i]));
        if (hammingDist == 0){
            return i; //This is the position at which we found the pat in the unlearning items
        }
        minDist = MIN(hammingDist,minDist);
    }
    return NOTFOUND;
}





/*
This funtion is used to generate the unlearning items stored in the training->unlearningPatts
*/

int generateUnlearningPatterns(netinfo * net, traininginfo * training){
    int i,dist,count;
    int cycling=0;
    count = 0;
    i = 0;
    training->numUnlearningPatts = 0;
    do{
        setNetworkRandom(net);
        dist = relaxNetwork(net); //number of cycles to relax to the pattern
        sortPatternInput(net,&(net->pat));
        if (net->pat.ratioStable < training->unlearningCutoffHigh){
        
            if (dist < net->maxCycles){  //If it is actually stable 
                if (training->uniqueUnlearningPatts){
                    if (compareToUnlearningPatterns(net,training,&(net->pat))==NOTFOUND){
                        if (training->removeLearntFromUnlearningPatts){
                            if (compareToLearntPatterns(net,training,&(net->pat))==NOTFOUND){
                                addUnlearningItem(net,training,&(net->pat));  // add if not found
                            }
                        } else {
                            addUnlearningItem(net,training,&(net->pat));
                    }
    
                    }
                } else {
                    if (training->removeLearntFromUnlearningPatts){
                        if (compareToLearntPatterns(net,training,&(net->pat))==NOTFOUND){
                            addUnlearningItem(net,training,&(net->pat));
                        }
                    } else {
                        addUnlearningItem(net,training,&(net->pat));
                    }
                }
            } else {
                cycling++;
            }
        }
        count++;
    }while((i<training->maxUnlearningPatts)&& (count < training->maxSamplesForUnlearningPatts));
    printf("%d %d ",cycling, count);
    return i;
  }

/*
 The process of unlearning is to:
 sample the network
 let it relax
 then do anti hebbian learning on the result.

 The interesting variables for unlearning are:
   the amount of unlearning to perform
   the type of cycles to use
 
 */
int unlearnCycles(netinfo * net, traininginfo * training, int cycles){
  int i,j,dist,cycling=0;
  int numActuallyUnlearnt = 0;
  FOR(i,cycles){
    FOR(j,training->unlearningCycleSize) {
      training->numUnlearningPatts = 0;
      setNetworkRandom(net);
      dist = relaxNetwork(net); //number of cycles to relax to the pattern
      sortPatternInput(net,&(net->pat));

      if (dist < net->maxCycles){  //If it is actually stable
        if (training->removeLearntFromUnlearningPatts){
          if (compareToLearntPatterns(net,training,&(net->pat))==NOTFOUND){
            addUnlearningItem(net,training,&(net->pat));
          }
        } else {
          addUnlearningItem(net,training,&(net->pat));
        }
      } else {
        cycling++;
      }
      /*
       Now we have an interesting choice.  What do we do with the pattern we just found
       There are three suggested situations
       1, always unlearn  = STDUnlearning
       2, unlearn only the items that are considered spurious
       2.1 Spurious by Ratio
       2.2 Spurious by mean
       3, just effect the heavily stable units (super heated)

       This last one actually need to be a part of the hebbian learning Partial
       procedure as this is where the changes are actually occuring.

       
           */
      switch(training->unlearningType){
      case unlearningStandard:
	hebbianLearnPartial(net, training, training->unlearningPatts, training->fixedOrder, 0, 1,
                            -training->unlearningConst); //note the negative sign on the learning constant
	numActuallyUnlearnt++;
	break;
      case unlearningSpuriousRatio:
	/*in this part we only unlearn the items who have a ratio below a given value
          "training->unlearningCutoffLow"*/
	//	printf("%lf < %lf",(double)net->pat.ratioStable,(double)training->unlearningCutoffLow);
	if (net->pat.ratioStable < training->unlearningCutoffLow){
	  hebbianLearnPartial(net, training, training->unlearningPatts, training->fixedOrder,0, 1,
			      -training->unlearningConst);
	  numActuallyUnlearnt++;
	}
	/*The rest fo the patterns are ignored for this unlearning condition
	 */
	break;
      case unlearningSpuriousMean:
	/*in this part we only unlearn the items who have a MEAN below a given value
	  "training->unlearningCutoffLow"*/
	if (net->pat.meanStable < training->unlearningCutoffLow){
	  hebbianLearnPartial(net, training, training->unlearningPatts, training->fixedOrder,0, 1,
			      -training->unlearningConst);
	  numActuallyUnlearnt++;
	}//end learning of low mean patterns
	break;
      case unlearningSpuriousOnly:
	if (compareToLearntPatterns(net, training, &(net->pat))==NOTFOUND){
	  hebbianLearnPartial(net, training, training->unlearningPatts, training->fixedOrder, 0, 1,
			      -training->unlearningConst);
	  numActuallyUnlearnt++;
	}
	break;
      }//end of switch statement

    }// end the FOR loop for learning cycle size
  }//end for loop for number of unlearning cyles
  printf("Number actually unlearnt:%d / %d = %6.3f (%d cycling)",numActuallyUnlearnt,cycles*training->unlearningCycleSize, numActuallyUnlearnt/(double)(cycles*training->unlearningCycleSize),cycling);
  return numActuallyUnlearnt;
};//end function


