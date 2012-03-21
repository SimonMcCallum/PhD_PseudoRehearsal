//                              -*- Mode: C -*- 
// net.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: trial.h,v 1.1 2002/10/20 11:19:43 simon Exp $
// 


#ifndef _trial_h_
#define _trial_h_


typedef struct multipTrialStorage
  {
    double ** weight;
    double * weightDeltaArray;
    double ** weightDelta;
  } trialStorage;

#endif
