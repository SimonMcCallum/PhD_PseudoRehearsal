#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "general.h"
#include "net.h"
#include "training.h"
#include "random.h"
#include "display.h"
#include "sampling.h"



double setPatternInput(netinfo * net, pattern * pat){
    int to,from;
    FOR(to,net->N){
        pat->unitInput[to] = 0;
        FOR(from,(net->N+(net->bias))){
            pat->unitInput[to] += net->weight[to][from] * pat->unit[from];
        }
    }
    return to;
}


/* This function rearranges the array passed so that the units are ordered 
such that the first learnt pattern is divided into ON followed by OFF
patterns.  This follows for all the learnt patterns.  This is done by
a modified quicksort
*/

int modQuickSort(netinfo * net, traininginfo * training, int * array ,int botPos, int topPos, int level){
    int swaps;
    int Rlow,Rhigh,Llow,Lhigh;
    
    swaps = 0;
    Llow = Lhigh = botPos;
    Rlow = Rhigh = topPos;/*not sure about the -1 but it works I'll work it out later*/
    
    while(Rlow > Lhigh){
        while((training->learntPatts[level].unit[array[Lhigh]] == net->ON)&&
                (Lhigh < Rlow))
            Lhigh++;
        while((training->learntPatts[level].unit[array[Rlow]] == net->OFF)&&
                (Lhigh < Rlow))
            Rlow--;
        if (Rlow > Lhigh){
           swap(array+Lhigh,array+Rlow);
            swaps++;
        }
    } //Now we have sorted by pattern Level.  now do next level.
    level++;
    if (level < training->maxLearntPatts) {
        if(((Lhigh-1)-(Llow))>0)swaps+=modQuickSort(net,training,array,Llow,Lhigh-1,level);
        if((Rhigh-(Rlow))>0)swaps+=modQuickSort(net,training,array,Rlow,Rhigh,level);
    }
    return swaps;
}

int sortAcrossMultiplePatterns(netinfo * net, traininginfo * training, int * order){
    return modQuickSort(net,training,order,0,net->N-1,0);
}



/*Note that this return interesting information through the patClose and patDist
*/
int checkAgainstLearntPatterns(netinfo * net, traininginfo * training, int cycleDist, int * patClose, int * patDist, int patInitChanges){
    int i,hammingDist,minDist=net->N;
    FOR(i,training->maxLearntPatts){
        hammingDist = equalState(net,&(net->pat),&(training->learntPatts[training->learntOrder[i]]));
        if (hammingDist == 0){
//            sortPatternInput(net,&(training->learntPatts[training->learntOrder[i]]));
            training->learntPatts[training->learntOrder[i]].frequency++;
            training->learntPatts[training->learntOrder[i]].cycleDist += cycleDist;
			training->learntPatts[training->learntOrder[i]].initialChanges += patInitChanges;
            return FOUND;
        }
        minDist = MIN(hammingDist,minDist);
        if (hammingDist == minDist) //this is the smallest one so far
            *(patClose) = i;
    }
    *(patDist) = minDist;
    return NOTFOUND;
}


int addStablePat(netinfo * net, traininginfo * training, pattern * pat, int cycleDist, int patClose, int patDist, int patInitChanges){
  int i;

  i = training->numSpuriousPatts;
    
    if (i >= training->maxSpuriousPatts)
        return NUMSTABLEFULL;
    copyPattern(net,&(training->spuriousPatts[i]),&(net->pat));
    sortPatternInput(net,&(training->spuriousPatts[i]));
    training->spuriousPatts[i].num = i;
    training->spuriousPatts[i].frequency = 1;
    training->spuriousPatts[i].cycleDist = cycleDist;
    training->spuriousPatts[i].minimumDistToLearntPattern = patDist;
    training->spuriousPatts[i].closestLearntPattern = patClose;
    training->spuriousPatts[i].initialChanges = patInitChanges;
    training->numSpuriousPatts++;
    return (training->numSpuriousPatts);
}


int checkAgainstOtherPatterns(netinfo * net, traininginfo * training,int cycleDist, int patInitChanges){
    int i;
    FOR(i,training->numSpuriousPatts){
      if (fastEqualState(net,&(net->pat),&(training->spuriousPatts[i])) == 0){
		training->spuriousPatts[i].frequency++;
		training->spuriousPatts[i].cycleDist += cycleDist;
		training->spuriousPatts[i].initialChanges += patInitChanges;
		return i;
      }
    }
    return NOTFOUND;
}


/* Currently not used and out of date
int testLearntPatt(netinfo * net, traininginfo * training){
    int i,dist,original,other;
    
    FOR(i,training->maxLearntPatts){
        training->learntPatts[i].frequency = 0;
        setPatternInput(net,&(training->learntPatts[i]));
        sortPatternInput(net,&(training->learntPatts[i]));
    }
    return i;    
}
*/    


void resetSamplingValues(netinfo * net, traininginfo * training){
    int i;
    training->numSpuriousPatts=0;
    FOR(i,training->maxLearntPatts){
        training->learntPatts[i].frequency = 0;
        training->learntPatts[i].cycleDist = 0;
        training->learntPatts[i].initialChanges = 0;
        setPatternInput(net,&(training->learntPatts[i]));
        sortPatternInput(net,&(training->learntPatts[i]));
    }
}

/*
This implements a noisy search for the level at which 50% of patterns
relax to the desired position.

double sampleBasinOfAttraction(netinfo * net, traininginfo * training, pattern * attractor){
  //double baseOverlap = 0.5;
  double currentOverlap = 1.0;
  double branchingFactor = 0.25;
  double rateOfChange = 0.25;
  int  dist;
  int success=1; //as we know that it is stable overlap at 1 give success
  int basinSamples = 10;
  double successThreshold = 0.5;
  double successCount =0;
  int i,j;
   
  success = 1;
  FOR(i,20){
    //printf("\r%5.2f",(i/(double)training->numCheck)*100);
    //fflush(stdout);
    currentOverlap = currentOverlap + rateOfChange;
    successCount = 0;
    FOR(j,basinSamples){
      setNetworkPartial(net,attractor,currentOverlap);
      dist = relaxNetwork(net); //number of cycles to relax to the pattern
      successCount += (fastEqualState(net,&(net->pat),attractor)==0);
    }
    branchingFactor *=0.8;
    rateOfChange = (0.5-(successCount/(double)basinSamples))*(branchingFactor); //ie 50% success
  }
  attractor->basinSize = currentOverlap;
  return currentOverlap;
}
*/
/*
This implements a binary search to find the level for which a 
set overlap amount of a stable pattern actually relaxes to the pattern.
This effectively test the basins of attraction.  
0.5 is the entire space relaxes to a pattern
0.75 still very large <- starting point.
0.875 reasonable size
0.9375 

I have removed this to try a non bianry search stratagy.
*/


double oldsampleBasinOfAttraction(netinfo * net, traininginfo * training, pattern * attractor){
  //double baseOverlap = 0.5;
  double currentOverlap = 1;
  double branchingFactor = 0.125;
  double successRate=1.0;
  int  dist;
  int basinSamples = 20;
  double successThreshold = 0.5;
  double successCount =0;
  double meanSuccess=0, meanFailure=0;
  int i,j;
   
  
  FOR(i,20){
    //printf("\r%5.2f",(i/(double)training->numCheck)*100);
    //fflush(stdout);
    //    if (successRate > successThreshold){ //still stable so ecrease overlap
	//  currentOverlap = currentOverlap - branchingFactor;
	//  } else {  //
	//  currentOverlap = currentOverlap + branchingFactor;
	//  }
    
    currentOverlap = currentOverlap +(((0.5 - successRate)+(SG(0.5 - successRate)*0.2))*branchingFactor);
	if (currentOverlap<0.0) currentOverlap=0.0;
    //printf("%6.4f <%6.4f> ",successRate,currentOverlap);
    successCount = 0;
    FOR(j,basinSamples){
      setNetworkPartial(net,attractor,currentOverlap);
      dist = relaxNetwork(net); //number of cycles to relax to the pattern
      successCount += (fastEqualState(net,&(net->pat),attractor)==0);
	if (successCount == 0){
		if(i>4) meanFailure+=currentOverlap;
	}else{
		if(i>4) meanSuccess+=currentOverlap;
	}
    }
    successRate = successCount/(double)basinSamples;
    branchingFactor *=0.95;
  }
  attractor->basinSize = currentOverlap;
  printf(" : %6.4f %5.3f %5.3f\n",currentOverlap,successRate,(meanSuccess+meanFailure)/(basinSamples*(20-5)) );
  return currentOverlap;
}


double sampleBasinOfAttraction(netinfo * net, traininginfo * training, pattern * attractor){
  //double baseOverlap = 0.5;
  double currentOverlap = 0.75;
  double branchingFactor = 0.025;
  double successRate=1.0;
  int  dist;
  int basinSamples = 250;
  int testingSamples = basinSamples-101;
  double successThreshold = 0.5;
  int success =0;
  int i;
  double maxOverlap=0.5,minOverlap=1.0;
  double meanSuccess=0, meanFailure=0;
  int numSuccess=0, numFailure=0;
   
  
  FOR(i,basinSamples){
    setNetworkPartial(net,attractor,currentOverlap);
    dist = relaxNetwork(net); //number of cycles to relax to the pattern
    success = (fastEqualState(net,&(net->pat),attractor)==0);
    
    branchingFactor *=0.98;
	if (success == 0){
		if (maxOverlap<currentOverlap){
			maxOverlap = currentOverlap;
		}
		if(i>testingSamples) {meanFailure+=currentOverlap; numFailure++;};
		currentOverlap += branchingFactor;
		if (currentOverlap > 1)
			currentOverlap = 1;
	}else{
		if (minOverlap>currentOverlap){
			minOverlap = currentOverlap;
		}
		if(i>testingSamples) {meanSuccess+=currentOverlap; numSuccess++;};
		currentOverlap -= branchingFactor;
		if (currentOverlap < 0.5)
			currentOverlap = 0.5;
	}
  }
   
  attractor->basinSize = (meanSuccess+meanFailure)/(basinSamples-testingSamples) ;
#ifdef _DEBUG  
  printf(" bs %9.6f %5.3f %5.3f %5.3f %5.3f %3d %3d <%8.6f> %8.6f %8.6f sb ",
	  currentOverlap,minOverlap,maxOverlap,meanSuccess/numSuccess, meanFailure/numFailure,
	  numSuccess,numFailure,
	  (meanSuccess+meanFailure)/((basinSamples-testingSamples)-1), 
	  ((meanSuccess/numSuccess)+(meanFailure/numFailure))/2.0,
	  currentOverlap-((meanSuccess+meanFailure)/((basinSamples-testingSamples)-1)));
#endif

 // oldsampleBasinOfAttraction(net, training, attractor);
  return (meanSuccess+meanFailure)/((basinSamples-testingSamples)-1) ;
}
 


int sampleNet(netinfo * net, traininginfo * training)
{
	int i,dist,position,patClose,patDist;
	int cycling=0,tooMany =0;

	resetSamplingValues(net,training);/*This is the place where the spurious stable patterns are reset
									  as well as the cycle distance and frequency*/
	fflush(stdout);
	FOR(i,training->numCheck){
		//printf("\r%5.2f",(i/(double)training->numCheck)*100);
		
		setNetworkRandom(net);
		dist = relaxNetwork(net); //number of cycles to relax to the pattern

		sortPatternInput(net,&(net->pat));
		if (dist < net->maxCycles){
			position = checkAgainstLearntPatterns(net,training,dist,&patClose,&patDist, net->pat.initialChanges);
			if (position == NOTFOUND){
				position = checkAgainstOtherPatterns(net,training,dist,net->pat.initialChanges);  //if it exists return the position else -1
				if (position == NOTFOUND){
					position = addStablePat(net,training,&(net->pat),dist,patClose,patDist,net->pat.initialChanges);
					if (position == NUMSTABLEFULL){ 
						tooMany++;
					} else {
						if(training->calcSpuriousBasins){
							sampleBasinOfAttraction(net,training,&(training->spuriousPatts[position-1]));
						}
					}

				}
			}
		} else {
			cycling++;
		}
	}
	printf("The number cycling is = %4d  the number of stable states not saved is %4d\n",cycling,tooMany);
	return i;
}

