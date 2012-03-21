//                              -*- Mode: C -*- 
// learning.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: pseudolearning.h,v 1.6 2003/06/16 07:40:00 simon Exp $
// 


#ifndef _pseudolearning_h_
#define _pseudolearning_h_

int addPseudoItem(netinfo * net, traininginfo * training, pattern * setPat);
int compareToPseudoPatterns(netinfo * net, traininginfo * training, pattern * setPat);

int generatePseudoPatterns(netinfo * net, traininginfo * training);

double deltaLearnPseudorehearsal(netinfo * net, traininginfo * training,int start, int end);

#endif
