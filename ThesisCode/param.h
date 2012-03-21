//                              -*- Mode: C -*- 
// sampling.h --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: param.h,v 1.7 2003/05/08 20:33:16 simon Exp $
// 


#ifndef _param_h_
#define _param_h_

#define NOT_FOUND -1

void initNetParam(netinfo * net);

void initTrainingParam(traininginfo * training);

int readParamFile(char *paramFilename,char * paramType);

int readNetParamFile(char *paramFilename);
int readTrainingParamFile(char *paramFilename);

int findParamPos(char *paramItem);
BOOL paramDefined(char *paramItem);
void paramError(char *paramItem);

int addParamFileData(char *paramItem, char *databuf, char * paramType);
int addParamFileLong(char *paramItem, long newLong, char * paramType);

int updateParamFileData(char *paramItem, char *databuf);
int updateParamFileLong(char *paramItem, long databuf);


void printParamDatabase();


void saveParamDatabase(char * filename, char * paramType);
void saveNetParamDatabase(char * filename);
void saveTrainParamDatabase(char * filename);


int getParamFileData(char *paramItem, char *databuf);
int getParamFileString(char *paramItem, char *newString);
int getParamFileToggle(char *paramItem, int *newToggle);
int getParamFileInt(char *paramItem, int *newInt);
int getParamFileLong(char *paramItem, long *newLong);
int getParamFileFloat(char *paramItem, double *newFloat);

int processNetParameters(netinfo * net);
int processTrainingParameters(traininginfo * training);


#endif

