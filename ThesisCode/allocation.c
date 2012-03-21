
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "general.h"
#include "net.h"
#include "training.h"
#include "allocation.h"

pattern * initPattern(netinfo*net,pattern * pat){
        memset(pat->unit 			,0,(net->N+1)*sizeof(double));
        memset(pat->unitInput 		,0,(net->N+1)*sizeof(double));
        memset(pat->unitNetInput	,0,(net->N+1)*sizeof(double));
        memset(pat->unitDelta 		,0,(net->N+1)*sizeof(double));
        memset(pat->unitError 		,0,(net->N+1)*sizeof(double));
        return pat;
}

pattern * allocPattern(netinfo*net,pattern * pat){
        pat->unit 			= (double *)calloc(net->N+1,sizeof(double));
        pat->unitInput 		= (double *)calloc(net->N+1,sizeof(double));
        pat->unitNetInput	= (double *)calloc(net->N+1,sizeof(double));
        pat->unitDelta 		= (double *)calloc(net->N+1,sizeof(double));
        pat->unitError 		= (double *)calloc(net->N+1,sizeof(double));
        pat->unit[net->N] = net->ON;
        return pat;
}


void freePattern(pattern * pat){
        free(pat->unit);
        free(pat->unitInput);
        free(pat->unitNetInput);
        free(pat->unitDelta);
        free(pat->unitError);
}

/********************************************************************************************************

*/
int initPatternMemory(netinfo * net, traininginfo * training){
    int i;
    FOR(i,training->maxLearntPatts){
        memset(training->learntPatts[i].unit 			,0,(net->N+1)*sizeof(double));
        memset(training->learntPatts[i].unitInput 		,0,(net->N+1)*sizeof(double));
        memset(training->learntPatts[i].unitNetInput	,0,(net->N+1)*sizeof(double));
        memset(training->learntPatts[i].unitDelta 		,0,(net->N+1)*sizeof(double));
        memset(training->learntPatts[i].unitError 		,0,(net->N+1)*sizeof(double));
    }

    FOR(i,training->maxSpuriousPatts){
        memset(training->spuriousPatts[i].unit 			,0,((net->N+1))*sizeof(double));
        memset(training->spuriousPatts[i].unitInput 		,0,(net->N+1)*sizeof(double));
        memset(training->spuriousPatts[i].unitNetInput 	,0,(net->N+1)*sizeof(double));
        memset(training->spuriousPatts[i].unitDelta 		,0,(net->N+1)*sizeof(double));
        memset(training->spuriousPatts[i].unitError 		,0,(net->N+1)*sizeof(double));
    }

    FOR(i,training->maxPseudoPatts){
      memset(training->pseudoPatts[i].unit 			,0,(net->N+1)*sizeof(double));
      memset(training->pseudoPatts[i].unitInput 		,0,(net->N+1)*sizeof(double));
      memset(training->pseudoPatts[i].unitNetInput 	,0,(net->N+1)*sizeof(double));
      memset(training->pseudoPatts[i].unitDelta 		,0,(net->N+1)*sizeof(double));
      memset(training->pseudoPatts[i].unitError 		,0,(net->N+1)*sizeof(double));
    }
    FOR(i,training->maxUnlearningPatts){
      memset(training->unlearningPatts[i].unit 			,0,(net->N+1)*sizeof(double));
      memset(training->unlearningPatts[i].unitInput 		,0,(net->N+1)*sizeof(double));
      memset(training->unlearningPatts[i].unitNetInput 	,0,(net->N+1)*sizeof(double));
      memset(training->unlearningPatts[i].unitDelta 		,0,(net->N+1)*sizeof(double));
      memset(training->unlearningPatts[i].unitError 		,0,(net->N+1)*sizeof(double));
    }
    return i;
}

 

int allocPatternMemory(netinfo * net, traininginfo * training){
    int i;

    training->learntPatts = (pattern *)calloc(training->maxLearntPatts,sizeof(pattern));
    FOR(i,training->maxLearntPatts){
        training->learntPatts[i].unit 			= (double *)calloc(net->N+1,sizeof(double));
        training->learntPatts[i].unitInput 		= (double *)calloc(net->N+1,sizeof(double));
        training->learntPatts[i].unitNetInput	= (double *)calloc(net->N+1,sizeof(double));
        training->learntPatts[i].unitDelta 		= (double *)calloc(net->N+1,sizeof(double));
        training->learntPatts[i].unitError 		= (double *)calloc(net->N+1,sizeof(double));
        training->learntPatts[i].unit[net->N] = net->ON;
    }

    training->spuriousPatts = (pattern *)calloc(training->maxSpuriousPatts,sizeof(pattern));
    FOR(i,training->maxSpuriousPatts){
        training->spuriousPatts[i].unit 			= (double *)calloc(net->N+1,sizeof(double));
        training->spuriousPatts[i].unitInput 		= (double *)calloc(net->N+1,sizeof(double));
        training->spuriousPatts[i].unitNetInput 	= (double *)calloc(net->N+1,sizeof(double));
        training->spuriousPatts[i].unitDelta 		= (double *)calloc(net->N+1,sizeof(double));
        training->spuriousPatts[i].unitError 		= (double *)calloc(net->N+1,sizeof(double));
        training->spuriousPatts[i].unit[net->N] = net->ON;
    }

    training->pseudoPatts = (pattern *)calloc(training->maxPseudoPatts,sizeof(pattern));
    FOR(i,training->maxPseudoPatts){
      training->pseudoPatts[i].unit 			= (double *)calloc(net->N+1,sizeof(double));
      training->pseudoPatts[i].unitInput 		= (double *)calloc(net->N+1,sizeof(double));
      training->pseudoPatts[i].unitNetInput 	= (double *)calloc(net->N+1,sizeof(double));
      training->pseudoPatts[i].unitDelta 		= (double *)calloc(net->N+1,sizeof(double));
      training->pseudoPatts[i].unitError 		= (double *)calloc(net->N+1,sizeof(double));
      training->pseudoPatts[i].unit[net->N] = net->ON;
    }
    training->unlearningPatts = (pattern *)calloc(training->maxUnlearningPatts,sizeof(pattern));
    FOR(i,training->maxUnlearningPatts){
      training->unlearningPatts[i].unit 			= (double *)calloc(net->N+1,sizeof(double));
      training->unlearningPatts[i].unitInput 		= (double *)calloc(net->N+1,sizeof(double));
      training->unlearningPatts[i].unitNetInput 	= (double *)calloc(net->N+1,sizeof(double));
      training->unlearningPatts[i].unitDelta 		= (double *)calloc(net->N+1,sizeof(double));
      training->unlearningPatts[i].unitError 		= (double *)calloc(net->N+1,sizeof(double));
      training->unlearningPatts[i].unit[net->N] = net->ON;
    }
    
    return i;
}

/********************************************************************************************************

*/
int allocNetMemory(netinfo * net){
    int i;
    allocPattern(net,&(net->pat));
    allocPattern(net,&(net->initPat));
    allocPattern(net,&(net->noisyPat));

    net->unitsToChange = (int*)calloc(net->N,sizeof(int));
    net->unitOrder = (int*)calloc(net->N,sizeof(int));
    net->noisyInput=(double*)calloc(net->N,sizeof(double));

    net->weightArray = (double*)calloc(net->N*(net->N+1),sizeof(double));
    net->weight 	 = (double**)calloc(net->N,sizeof(double*));
    FOR(i,net->N){
        net->weight[i] = net->weightArray+(i*(net->N+1));
    }

    net->weightDeltaArray = (double*)calloc(net->N*(net->N+1),sizeof(double));
    net->weightDelta = (double**)calloc(net->N,sizeof(double*));
    FOR(i,net->N){
        net->weightDelta[i] = net->weightDeltaArray+(i*(net->N+1));
    }
    return i;
}
