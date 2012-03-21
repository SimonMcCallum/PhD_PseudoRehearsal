// param.c --- 
// Author          : Simon McCallum
// Last Modified By: Simon McCallum
// Update Count    : 1
// Status          : Unknown, Use with caution!
// $Id: param.c,v 1.28 2003/06/14 09:25:21 simon Exp $
// 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#include "general.h"
#include "net.h"
#include "training.h"
#include "param.h"

#define STR_EQ(x,y) !strncmp((x),(y),MIN(strlen(y),strlen(x)))

// The maximum size of the param database.
#define PARAM_DATABASE_SIZE         300

// The maximum length of a param string.
#define PARAM_MAXIMUM_LENGTH        100

// The maximum length of the param data strings.
#define PARAM_DATA_MAXIMUM_LENGTH   100

// Global storage for the param database.
typedef struct {
	char param[PARAM_MAXIMUM_LENGTH];
	char data[PARAM_DATA_MAXIMUM_LENGTH];
	char paramType[10];
} param_entry;

int numParams=0;
param_entry paramDatabase[PARAM_DATABASE_SIZE];



/* There are a lot of params in this program.  The initialisation of these values
is done in this init param function.  The idea is that having set these up as the most
simple values possible the program will read the two param files.  Net and training.

*/
void initNetParam(netinfo * net){

	net->N =   400; // Lets start small and without much calculation
	net->pat.num =0;
	net->pat.unit=NULL;
	net->pat.unitInput=NULL;
	net->pat.unitNetInput=NULL;
	net->pat.unitDelta=NULL;
	net->pat.unitError=NULL;
	net->pat.sum=0; //of units on
	net->pat.cycleDist=0;
	net->pat.cycle=0;
	net->pat.leastStable=0;
	net->pat.stable=0;
	net->pat.meanStable=0;
	net->pat.mostStable=0;
	net->pat.stddev=0;
	net->pat.coefficientOfVariation=0;
	net->pat.ratioStable=0;  
	net->pat.meanUnitInput=0;

	net->pat.frequency=0;
	net->pat.minimumDistToLearntPattern=0;
	net->pat.closestLearntPattern=0; //This is the current pattern in the Network.
	net->pat.initialChanges=0;


	net->unitsInRatio = 2; 

	net->weightArray = NULL;
	net->weight = NULL;
	net->weightDeltaArray=NULL;
	net->weightDelta=NULL;


	net->ON = 1;
	net->OFF = -1;
	net->maxCycles = 1;
	net->activationType = 0;
	net->bias = 0;    
	net->learnTiming = ONLINE;
	net->relaxation = ASYNCHRONOUS;
	net->symmetricWeights = TRUE;
	net->calcHammingDistRelax = 0;
	net->calcInitialChanges =FALSE;
	net->calcWeightAverages =FALSE;

	net->thermalDelta=FALSE;
	net->startTemperature=128.0;
	net->temperature =net->startTemperature;

	net->noiseDuringRelaxation = FALSE;
	net->gaussianNoiseType=0;
	net->lenghtOfTimeStableInRelaxation=2;
	net->gaussianRelativeRangeForRelaxation=0;
	net->gaussianAbsoluteRangeForRelaxation=0;

	net->weightCappingType=0; // choises are NOCAP, HARDCAP, SOFTCAP
	net->weightCappingValue=1; // the point at which to cap weights;

	net->weightDecay=0; // True of false; 
	net->weightDecayValue=0.1; //10% ;

	net->weightNormalisationType=0; // choises are NONORMALISATION,NORMAILSENET,NORMALISEUNIT,NORMALISEUNIT_ZEROMEAN,NORMALISEUNIT_BOTH
	net->weightNormalisationValue = 1; // the point at which to cap weights;

	net->randomSeed=0; /*This is saved so that we can rerun simulations. if it is present then it will be used.  If not a new random number will be created and saved with the param file.*/

	net->sampleProbOn = 0.5; //The probability of a unit begin on during random sampling of patterns
	//must check with generateProbOn in training.  

	net->averageInput = net->N;

	net->gnuplotGraphs = 0;


	net->debugLevel=0;
}

void initTrainingParam(traininginfo * training){
	training->numTrials = 50;
	training->learningConst = 0.1;	
	training->learningConstDecay = 	training->learningConst;
	training->learningConstDecayTo =0.1;
	training->momentum      = 0.9;


	training->maxLearntPatts = 10; // The number of patterns to learn generally set at the max value with all cycles
	training->numCheck =4;       // The sampling size when checking the network
	training->maxSpuriousPatts=4; //maximum number of patterns to store as Stable Spurious pattern
	training->maxPseudoPatts=4; //maximum number of patterns to store as Pseudo patterns
	training->maxSamplesForPseudoPatts=1; //How many sample to try to get the appropriate number of pseudo items
	training->maxUnlearningPatts=50; //maximum number of patterns to store as Pseudo patterns
	training->maxSamplesForUnlearningPatts=1000; //How many sample to try to get the appropriate number of pseudo items


	training->unlearningCutoffHigh=1;
	training->unlearningCutoffLow=1;
	training->unlearningCycleSize=50;
	training->unlearningConst = 0.01;//note that this is the value not a negative value
	training->unlearningType = unlearningStandard;
	training->unlearningSuperHeated = FALSE;


	training->uniqueUnlearningPatts = FALSE;
	training->removeLearntFromUnlearningPatts = FALSE;

	training->pseudoRehearsalCutoff=0.3333; // this is the cutoff for inclusion in the pseudo population.
	training->useMaxLowestLearntvsCutoff = FALSE; // evaluate the lowest ratio while learning and max with cut off num. 
	
	training->realRehearsal=FALSE;
	training->realRehearsalPercent=1.0;
	training->includePseudoInError=FALSE;
	training->noiseOnPseudoPatts=FALSE;
	training->noiseRatioPseudoPatts=1.0;
	training->learningRatioPseudoPatts=1.0;

	training->allEpochLearning=FALSE;
	training->trainingEpochs=1;    //number of epoachs to train
	training->initialTrainingEpochs=1;    //number of epoachs to train
	training->errorCriteria=0;  //error criteria while traing
	training->errorTail=0;      //decay of the error value creating an error tail

	training->hebbianCycles=1;  // number of cycles used for hebbian learning
	training->hebbianNoiseLevel=0; // noise level while training


	//Binary choice modal types:
	training->noiseHebbianNegative=0;
	//This is the parameter that deterimines what the noise during hebbian learning is actually goint to do.

	//Modal types:
	training->learningType  = HEBBIAN;   //one of STRICT DELTA, HEBBIAN, PSEUDODELTA 
	training->errorCalcType = SIGMOIDERROR; //SIGMOIDERROR , DELTAERROR the type of error.


	//Instance Variables
	training->numSpuriousPatts=0;	// varies to show that actual number of patterns are stable currently
	training->numPseudoPatts=0; //the current number of PseuoPatts

	training->step=1;
	training->initialNumberPatts=1;
	training->numLearntPatts=0;


	training->uniquePseudoPatts=0;  // this makes sure that a pseudo item only appears once in an epoach
	training->pseudoRehearsalReal=FALSE;
	training->removeLearntFromPseudoPatts=0; // this makes sure that the learnt patterns are not part of the pseudo patterns.  This is not a useful thing to do except during testing to see if PR actually works with this restriction.

	training->adjustLearningForPobability=0;

	training->noiseOnInput=0; //T or F noise on the input of units during training
	training->noiseOnOutput=0; // T or F noise on the output of units during training
	training->noiseHetroAssociative=0; // T or F map a noisy pattern to a real pattern.

	training->hetroNoiseLevel=0; //percent chance of unit fkipping

	training->gaussianRelativeRange=0;
	training->gaussianAbsoluteRange=0;
	training->gaussianNoiseType=0; // ABSOLUTE , RELATIVE (to the mean absolute input value)


	training->generateProbOn=0.5; //The probability of a unit begin on in the random generation of patterns
	training->generateActivationExactLevel=0;

	training->displaySpuriousPatts=0;
	training->displaySpuriousPatternDetails=0;

	training->reorderUnitsinPatts=0; /* this is the param that tells us to
									 reorder the units in the patterns so that they follow the pattern
									 ########........ then ####....####.... then ##..##..##..##.. etc*/
	training->calcSpuriousBasins=0; 
	training->calcLearntBasins=1; 



	training->saveStabilityProfilePatternCheck=0;      
	training->numSpuriousPatsProfiles =0;
	//Display check boxes
	training->displayTestedPatterns=1;

	training->displayLearntPattUnits=1;
	training->displayLearntPattInput=0;
	training->displayLearntPattNetInput=0;
	training->displayLearntPattSummary=1;

	training->displaySpuriousPattUnits=1;
	training->displaySpuriousPattInput=0;
	training->displaySpuriousPattNetInput=0;
	training->displaySpuriousPattSummary=1;

	training->displayPseudoPattUnits=1;
	training->displayPseudoPattInput=0;
	training->displayPseudoPattNetInput=0;
	training->displayPseudoPattSummary=1;

	strcpy(training->patternFilename,"none");

	training->learntPatts=NULL; //the Patterns that are to be learnt
	training->spuriousPatts=NULL; //Patterns that are stable
	training->pseudoPatts=NULL; //Storage for the pseudo patterns, used in PR.
}










/* Read in a paramFile. Passed a filename for the paramFile.
returns the number of parameters read into the database
*/
int readParamFile(char *paramFilename, char * paramType){
	FILE *paramFile;
	int oldParamNumber=numParams;
	char tempbuf[MAX_STRING_LENGTH];
	int i,j;

	if (paramFilename == NULL) {
		// There is no param file so no more parameters too add
		return oldParamNumber-numParams;
	}
	printf("Reading File <%s>\n", paramFilename);
	// Open the paramFile, or give an error.
	paramFile = fopen(paramFilename, "r");
	if (paramFile == NULL) {
		sprintf(tempbuf, "can't open paramFile <%s>\n", paramFilename);
		perror(tempbuf);
		exit(-1);
	}

	// Read in lines from the param file into the param database.

	do {
		// Read a line; NULL is returned on EOF or error.
		if (fgets(tempbuf, MAX_STRING_LENGTH, paramFile) == NULL)
			break;

		// Only parse lines that have a param on them and that don't begin
		// with a # symbol.
		if ((sscanf(tempbuf, "%s", paramDatabase[numParams].param) == 1) &&
			(paramDatabase[numParams].param[0] != '#')) {
				strcpy(paramDatabase[numParams].paramType,paramType);
				// Strip trailing newline from line if it exists.
				if (tempbuf[strlen(tempbuf) - 1] == '\n')
					tempbuf[strlen(tempbuf) - 1] = '\0';

				// if the param starts with whitespace
				// Scan through the whitespace looking for the first non-whitespace
				// characters or end-of-string.
				i=0;
				while((tempbuf[i] == ' ') || (tempbuf[i] == '\t') ||
					(tempbuf[i] == '\n')){
						i++;
				};
				// Scan through text in tempbuf looking for the next whitespace character.
				while ((tempbuf[i] != ' ') && (tempbuf[i] != '\t') &&
					(tempbuf[i] != '\n') && (tempbuf[i] != '\0')){
						i++;
				};

				// Scan through the whitespace looking for the first non-whitespace
				// character or end-of-string.
				while((tempbuf[i] == ' ') || (tempbuf[i] == '\t') ||
					(tempbuf[i] == '\n')){
						i++;
				}
				// If we didn't find any non-whitespace chars before the end of the
				// string then there wasn't a data item on that line.
				if (tempbuf[i] == '\0') {
					//This would be a blank line.  I will let this happen
				} else {

					// Check the length restrictions
					if (strlen(tempbuf) - i > PARAM_DATA_MAXIMUM_LENGTH) {
						sprintf(tempbuf, "data %s too long (max %d)\n", &tempbuf[i],
							PARAM_DATA_MAXIMUM_LENGTH);
						perror(tempbuf);
						perror("Param name too long\n");
					} else {


						// Copy the param data item from tempbuf at position i into
						// the param database. If it is not new.
						// Search through param database looking for the right param item.
						for (j = 0; (j < numParams) && (strncmp(paramDatabase[numParams].param,
							paramDatabase[j].param,
							MAX(strlen(paramDatabase[numParams].param),
							strlen(paramDatabase[j].param))) != 0); j++)
							;
						//printf("%2d, %s\n",j,paramDatabase[j].param);
						strcpy(paramDatabase[j].data, &tempbuf[i]);
						if(j==numParams){
							numParams++;
						}
					}
				}

				// Check for database overruns.
				if (numParams >= PARAM_DATABASE_SIZE) {
					sprintf(tempbuf, "too many param items (max %d)\n", PARAM_DATABASE_SIZE);
					perror(tempbuf);
				}
		}
		// Keep reading params forever. Break out of this loop when fgets()
		// returns NULL (indicating EOF).
	} while (TRUE);

	fclose(paramFile);
	printf(">> Param file %s read: %d items.\n", paramFilename, numParams);

	return (numParams - oldParamNumber); //return the number of new params
}

/*reading the different types of param files so they can be saved to the correct type of file*/
int readNetParamFile(char * filename){
	char * newFilename;
	if (strncmp(".net",filename+(strlen(filename)-4),4)){
		newFilename = malloc(sizeof(char)*MAX_STRING_LENGTH);
		strcpy(newFilename,filename);
		strcat(newFilename,".net");
	} else {
		newFilename = filename;
	}
	return readParamFile(newFilename,"net");
}

int readTrainingParamFile(char * filename){
	char * newFilename;
	if (strncmp(".train",filename+(strlen(filename)-6),6)){
		newFilename = malloc(sizeof(char)*MAX_STRING_LENGTH);
		strcpy(newFilename,filename);
		strcat(newFilename,".train");
	} else {
		newFilename = filename;
	}
	return readParamFile(newFilename,"train");
};


/* returns the location of the parameter or NOT_FOUND (-1) if not found*/
int findParamPos(char *paramItem){
	int j;
	for (j = 0; (j < numParams) && (strcmp(paramItem,
		paramDatabase[j].param) != 0); j++){
			//no-op
	}

	if (j == numParams){
		return NOT_FOUND;
	} else {
		return j;
	}
}

BOOL paramDefined(char *paramItem){
	return(findParamPos(paramItem)!=NOT_FOUND);
}

void paramError(char *paramItem){  
	char junk[255];
	printf("\nWARNING option not found %s\n",paramItem);
	fprintf(stderr,"\nstderr:WARNING option not found %s\n",paramItem);
	scanf("%s",junk);
	if (junk[0]!='y') {
		exit(-1);
	}

}

int addParamFileData(char *paramItem, char *databuf, char * paramType){
	int pos;
	pos = findParamPos(paramItem);       
	if (pos == NOT_FOUND){//add a new param
		strcpy(paramDatabase[numParams].param,paramItem);
		strcpy(paramDatabase[numParams].paramType,paramType);
		strcpy(paramDatabase[numParams].data, databuf);
		numParams++;
		return numParams;
	} else { //update value
		return -1;
	}
};

int addParamFileLong(char *paramItem, long newLong, char * paramType){
	char * databuf;
	databuf = (char *)malloc(sizeof(char)*256);
	sprintf(databuf,"%ld",newLong);
	return addParamFileData(paramItem,databuf,paramType);
};

int updateParamFileData(char *paramItem, char *databuf){
	int pos;
	pos = findParamPos(paramItem);       
	if (pos == NOT_FOUND){//add a new param
		return NOT_FOUND;
	} else { //update value
		strcpy(paramDatabase[pos].data, databuf);
		return pos;
	}
};

int updateParamFileLong(char *paramItem, long newLong){
	char databuf[10];
	sprintf(databuf,"%ld",newLong);
	return updateParamFileData(paramItem,databuf);
};


void printParamDatabase(){
	int i;
	for(i=0; i< numParams; i++){
		printf("%30s %s\n",paramDatabase[i].param, paramDatabase[i].data);
	}
}

void saveParamDatabase(char * filename, char * paramType){
	int i;
	FILE * saveParamFile;

	if (NULL == (saveParamFile = fopen(filename,"w"))){
		perror("cannot open param save file:\n");
	} else {

		for(i=0; i< numParams; i++){
			if(STR_EQ(paramDatabase[i].paramType,paramType)){
				fprintf(saveParamFile,"%30s %s\n",paramDatabase[i].param, paramDatabase[i].data);
			}
			//printf("%s,%s\n",paramDatabase[i].paramType,paramType);
		}
		fclose(saveParamFile);
	}

}

/* save the Params related to the network.  This will add .net to the filename if it does not currently
have .net on the end of the filename*/
void saveNetParamDatabase(char * filename){
	char * newFilename;
	if (strncmp(".net",filename+(strlen(filename)-4),4)){
		newFilename = malloc(sizeof(char)*MAX_STRING_LENGTH);
		strcpy(newFilename,filename);
		strcat(newFilename,".net");
	} else {
		newFilename = filename;
	}
	saveParamDatabase(newFilename,"net");
}


void saveTrainParamDatabase(char * filename){
	char * newFilename;
	if (strncmp(".train",filename+(strlen(filename)-4),4)){
		newFilename = malloc(sizeof(char)*MAX_STRING_LENGTH);
		strcpy(newFilename,filename);
		strcat(newFilename,".train");
	} else {
		newFilename = filename;
	}
	saveParamDatabase(newFilename,"train");
}
/****************************************************************************************
This changes the array pointed to by databuf to the string of characters starting with the first non space character in the parameter data to the end of line or the // comment.  The function is used by the other getParamFile* functions as the sorce of the string to process
*/
int getParamFileData(char *paramItem, char *databuf)
{
	int i,j;
	int startComment, foundComment = FALSE;

	// Search through param database looking for the right param item.
	for (i = 0; (i < numParams) && (strncmp(paramItem, paramDatabase[i].param,strlen(paramItem)) != 0); i++)
		;

	// Did we reach the end without finding the param? If so, return FALSE.
	if (i == numParams){
		return FALSE;
	} else {
		j=0;
		startComment = strlen(paramDatabase[i].data);
		while((j+1 < strlen(paramDatabase[i].data))&&!foundComment) {
			if (!strncmp(paramDatabase[i].data+(j),"//",2)){
				startComment = j;
				foundComment=TRUE;
			}
			j++;
		}
		strncpy(databuf, paramDatabase[i].data,startComment);
		databuf[startComment]='\0';
		return TRUE;
	}
}

int getParamFileString(char *paramItem, char *newString)
{
	int findSuccess;
	char tempbuf[PARAM_DATA_MAXIMUM_LENGTH];

	// Look for the param and return failure if necessary
	findSuccess = getParamFileData(paramItem, tempbuf);
	if (!findSuccess){
		return FALSE;
	} else {
		return (sscanf(tempbuf,"%s",newString)==1);
	}
}


int getParamFileToggle(char *paramItem, int *newToggle)
{
	int findSuccess;
	char tempbuf[PARAM_DATA_MAXIMUM_LENGTH];

	// Look for the param and return failure if necessary
	findSuccess = getParamFileData(paramItem, tempbuf);
	if (!findSuccess)
		return FALSE;

	if (STR_EQ(tempbuf, "on")||(STR_EQ(tempbuf, "TRUE"))||(STR_EQ(tempbuf, "yes"))||(STR_EQ(tempbuf, "1"))) {
		*newToggle = 1;
		return TRUE;
	} else {
		if ((STR_EQ(tempbuf, "off")) || (STR_EQ(tempbuf, "FALSE"))||(STR_EQ(tempbuf, "no")) ||(STR_EQ(tempbuf, "0"))) {
			*newToggle = 0;
			return TRUE;
		} else {
			sprintf(tempbuf, "incorrect toggle value for param %s\n", paramItem);
			perror(tempbuf);
			return FALSE;
		}
	}
}

int getParamFileInt(char *paramItem, int *newInt)
{
	int findSuccess;
	double multiplier,addition;
	int base;
	char tempbuf[PARAM_DATA_MAXIMUM_LENGTH];
	char otherParam[PARAM_MAXIMUM_LENGTH];
	int i;
	// Look for the param and return failure if necessary
	findSuccess = getParamFileData(paramItem, tempbuf);
	if (!findSuccess)
		return FALSE;
	// This is where I look for params that are multiplied values of other params
	//This is very useful for changing values that relate to other values
	for(i=0;(i<strlen(tempbuf))&&(tempbuf[i]!='*')&&(tempbuf[i]!='+');i++)
		;
	if (i < strlen(tempbuf)){
		if(tempbuf[i]=='*'){
			sscanf(tempbuf,"%lf*%s",&multiplier,otherParam);
			getParamFileInt(otherParam,&base);
			*(newInt) = (int)(multiplier * base);
		}else{
			sscanf(tempbuf,"%lf+%s",&addition,otherParam);
			getParamFileInt(otherParam,&base);
			*(newInt) = (int)(addition + base);
		}
	} else {
		if (sscanf(tempbuf, "%d", newInt) == 1){
			return TRUE;
		} else {
			sprintf(tempbuf, "corrupt number value for param %s\n", paramItem);
			perror(tempbuf);
			return FALSE;
		}
	}
	return TRUE;
}

int getParamFileList(char *paramItem, int *list)
{
	int findSuccess;
	int base;
	char * tempbuf;
	char otherParam[PARAM_MAXIMUM_LENGTH];
	int i,pos;

	tempbuf = (char*)calloc(PARAM_DATA_MAXIMUM_LENGTH,sizeof(char));
	// Look for the param and return failure if necessary
	findSuccess = getParamFileData(paramItem, tempbuf);
	if (!findSuccess)
		return FALSE;
	// This is where I look for params that are multiplied values of other params
	//This is very useful for changing values that relate to other values
	i=0;
	while ((0<strlen(tempbuf))&&(sscanf(tempbuf, "%d", &pos) == 1)&&(tempbuf[0]!='/')){
		list[pos]=TRUE;
		i++;
		while((0<strlen(tempbuf))&&(tempbuf[0]>='0')&&(tempbuf[0]<='9')){
			tempbuf=tempbuf+1;
		}
		while((0<strlen(tempbuf))&&((tempbuf[0]==' ')||(tempbuf[0]==','))){
			tempbuf=tempbuf+1;
		}

		fflush(stdout);
	}
//	printf(" list %d %d %d %d\n",list[1],list[2],list[3],list[4]);
	return (i>0)?TRUE:FALSE;
}



int getParamFileLong(char *paramItem, long *newLong)
{
	int findSuccess;
	double multiplier,addition;
	int base;
	char tempbuf[PARAM_DATA_MAXIMUM_LENGTH];
	char otherParam[PARAM_MAXIMUM_LENGTH];
	int i;
	// Look for the param and return failure if necessary
	findSuccess = getParamFileData(paramItem, tempbuf);
	if (!findSuccess)
		return FALSE;
	// This is where I look for params that are multiplied values of other params
	//This is very useful for changing values that relate to other values
	for(i=0;(i<strlen(tempbuf))&&(tempbuf[i]!='*')&&(tempbuf[i]!='+');i++)
		;
	if (i < strlen(tempbuf)){
		if(tempbuf[i]=='*'){
			sscanf(tempbuf,"%lf*%s",&multiplier,otherParam);
			getParamFileInt(otherParam,&base);
			*(newLong) = (long)(multiplier * base);
		}else{
			sscanf(tempbuf,"%lf+%s",&addition,otherParam);
			getParamFileInt(otherParam,&base);
			*(newLong) = (long)(addition + base);
		}
	} else {
		if (sscanf(tempbuf, "%ld", newLong) == 1){
			return TRUE;
		} else {
			sprintf(tempbuf, "corrupt number value for param %s\n", paramItem);
			perror(tempbuf);
			return FALSE;
		}
	}
	return TRUE;
}

int getParamFileFloat(char *paramItem, double *newFloat)
{
	int findSuccess;
	char tempbuf[PARAM_DATA_MAXIMUM_LENGTH];
	double multiplier;
	double base;
	char otherParam[PARAM_MAXIMUM_LENGTH];
	int i;

	// Look for the param and return failure if necessary
	findSuccess = getParamFileData(paramItem, tempbuf);
	if (!findSuccess)
		return FALSE;
	for(i=0;(i<strlen(tempbuf))&&(tempbuf[i]!='*');i++)
		;
	if (i < strlen(tempbuf)){
		sscanf(tempbuf,"%lf*%s",&multiplier,otherParam);
		getParamFileFloat(otherParam,&base);
		*(newFloat) = (int)(multiplier * base);
	}else{

		if (sscanf(tempbuf, "%lf", newFloat) == 1){
			return TRUE;
		} else {
			sprintf(tempbuf, "corrupt float value for param %s\n", paramItem);
			perror(tempbuf);
			return FALSE;
		}
	}
	return FALSE;
}



int processNetParameters(netinfo * net)
{
	char tempbuf[PARAM_DATA_MAXIMUM_LENGTH];

	getParamFileInt("N",&(net->N)); // = 32; //Number of units in the network


	getParamFileData("activationType",tempbuf);
	if (STR_EQ(tempbuf,"THRESHOLD")){
		net->activationType = THRESHOLD;  
	} else {
		if (STR_EQ(tempbuf,"SIGMOID")){
			net->activationType = SIGMOID;
		}else{
			paramError("activationType");
		}
	}

	//The activation type of the network
	/*For threshold we need to define the output of the unit when above and 
	below threshold  these are*/
	getParamFileFloat("ON",&(net->ON)); //= 1;  
	getParamFileFloat("OFF",&(net->OFF)); // = -1;

	/*When relaxing the network the maximum number of cycles to relax*/
	getParamFileInt("maxCycles",&(net->maxCycles)); // = 2*net->N;
	/* Form of relaxation*/
	getParamFileData("relaxation",tempbuf);
	if (STR_EQ(tempbuf,"ASYNCHRONOUS")){
		net->relaxation = ASYNCHRONOUS;
	}else{ 
		if (STR_EQ(tempbuf,"SYNCHRONOUS")){
			net->relaxation = SYNCHRONOUS;
		}else{
			paramError("relaxation");
		}
	}


	/*Calculating the ratio using a number of units as a percent of N the 
	number of units*/
	getParamFileFloat("unitsInRatio",&(net->unitsInRatio)); //= 0.1; //ie 0.1 = 10%

	/*Is there a bias unit*/
	getParamFileToggle("bias",&(net->bias)); // =0; //TRUE;

	/* What type of learning batch or online*/
	getParamFileData("learnTiming",tempbuf);
	if (STR_EQ(tempbuf,"ONLINE")){
		net->learnTiming = ONLINE;
	}else{
		if (STR_EQ(tempbuf,"BATCH")){
			net->learnTiming = BATCH;
		}else{
			paramError("learnTiming");
		}
	}

	/* This forces the connections to be symmetric */
	getParamFileToggle("symmetricWeights",&(net->symmetricWeights)); // 0; //TRUE;

	getParamFileData("weightCappingType",tempbuf);
	if(STR_EQ(tempbuf,"NOCAP")){
		net->weightCappingType = NOCAP;
	} else {
		if(STR_EQ(tempbuf,"HARDCAP")){
			net->weightCappingType = HARDCAP;
		} else {
			if(STR_EQ(tempbuf,"SOFTCAP")){
				net->weightCappingType = SOFTCAP;
				//NOCAP , HARDCAP, SOFTCAP
			} else {
				paramError("weightCappingType");
			}
		}
		getParamFileFloat("weightCappingValue",&(net->weightCappingValue)); //= 4;
	}
	getParamFileToggle("weightDecay",&(net->weightDecay)); // = FALSE; // 
	if (net->weightDecay) {
		getParamFileFloat("weightDecayValue",&(net->weightDecayValue)); // = 0.1; //10% ie weight*(1-val)
	}

	getParamFileData("weightNormalisationType",tempbuf);
	if (STR_EQ(tempbuf,"NONORMALISATION")){ 
		net->weightNormalisationType = NONORMALISATION;
	} else {
		//choises are NONORMALISATION,NORMALISENET,NORMALISEUNIT, NORMALISEUNIT_ZEROMEAN, NORMALISEUNIT_BOTH
		if (STR_EQ(tempbuf,"NORMALISENET")){ 
			net->weightNormalisationType = NORMALISENET;
		} else {
			if (STR_EQ(tempbuf,"NORMALISEUNIT_ZEROMEAN")){ 
				net->weightNormalisationType = NORMALISEUNIT_ZEROMEAN;
			} else {
				if (STR_EQ(tempbuf,"NORMALISEUNIT_BOTH")){ 
					net->weightNormalisationType = NORMALISEUNIT_BOTH;
				} else {
					if (STR_EQ(tempbuf,"NORMALISEUNIT")){ 
						net->weightNormalisationType = NORMALISEUNIT;
					} else {
						paramError("weightNormalisationType");
					}
				}
			}
		}
		getParamFileFloat("weightNormalisationValue",&(net->weightNormalisationValue)); //
		// = 5*net->N; // the point at which to normalise;
	}

	getParamFileToggle("thermalDelta",&(net->thermalDelta)); // = FALSE;
	getParamFileFloat("startTemperature",&(net->startTemperature)); //


	getParamFileToggle("noiseDuringRelaxation",&(net->noiseDuringRelaxation)); // = FALSE;
	getParamFileToggle("calcHammingDistRelax",&(net->calcHammingDistRelax)); // = FALSE;
	getParamFileToggle("calcInitialChanges",&(net->calcInitialChanges)); // = FALSE;

	if(net->noiseDuringRelaxation){
		getParamFileData("gaussianNoiseType",tempbuf);
		if (STR_EQ(tempbuf,"ABSOLUTE")){
			net->weightNormalisationType = ABSOLUTE;
			getParamFileFloat("gaussianAbsoluteRangeForRelaxation",&(net->gaussianAbsoluteRangeForRelaxation));
		} else {
			if (STR_EQ(tempbuf,"RELATIVE")){
				net->gaussianNoiseType = RELATIVE; //Remember to check training for noise type;
				getParamFileFloat("gaussianRelativeRangeForRelaxation",&(net->gaussianRelativeRangeForRelaxation));
			} else {
				paramError("gaussianNoiseType");
			}
		}
	}
	getParamFileInt("lenghtOfTimeStableInRelaxation",&(net->lenghtOfTimeStableInRelaxation));
	getParamFileLong("randomSeed",&(net->randomSeed));
//	getParamFileInt("randomSeed",&(net->randomSeed));

	getParamFileFloat("sampleProbOn",&(net->sampleProbOn)); // = 0.5; //This is the proability of a unit being on during sampling

	getParamFileToggle("gnuplotGraphs",&(net->gnuplotGraphs)); // = 1; //TRUE;


	/* The debugging level  0 is no debugging */
	getParamFileInt("debugLevel",&(net->debugLevel)); // = 1;
	return 1;
}





int processTrainingParameters(traininginfo * training){
	char tempbuf[PARAM_DATA_MAXIMUM_LENGTH];
	printf("\n");

	getParamFileInt("numTrials",&(training->numTrials)); //  learning scalar

	getParamFileFloat("learningConst",&(training->learningConst)); //  learning scalar
	training->learningConstDecayTo=training->learningConst;
	training->learningConstDecay=training->learningConst;
	getParamFileFloat("learningConstDecayTo",&(training->learningConstDecayTo)); //  learning decay over epochs
	getParamFileFloat("momentum",&(training->momentum)); //momentum

	getParamFileInt("maxLearntPatts",&(training->maxLearntPatts)); // = 5;
	getParamFileToggle("patternsFromFile",&(training->patternsFromFile));
	if(training->patternsFromFile){
		getParamFileString("patternFilename",training->patternFilename);
	}
	getParamFileInt("trainingEpochs",&(training->trainingEpochs)); // = 1000;
	training->initialTrainingEpochs = training->trainingEpochs;
	getParamFileInt("initialTrainingEpochs",&(training->initialTrainingEpochs)); // = 1000;
	getParamFileToggle("allEpochLearning",&(training->allEpochLearning)); // = FALSE;
	/* was at 1 */
	getParamFileFloat("hebbianNoiseLevel",&(training->hebbianNoiseLevel)); // = 0.0; // This is the noise when training the network with hebbian learning
	getParamFileInt("hebbianCycles",&(training->hebbianCycles)); // = 1; // Number of cycles to learn the set of patterns with hebbian learning
	getParamFileToggle("noiseHebbianNegative",&(training->noiseHebbianNegative)); // = FALSE;
	/* end 1*/
	/* was at 2 */
	getParamFileFloat("pseudoRehearsalCutoff",&(training->pseudoRehearsalCutoff)); // = 0.3;
	/*this is the cut off that prevents spurious states from being included as pseudo items.
	0.0 for this values means no cutoff at all.*/
	getParamFileToggle("useMaxLowestLearntvsCutoff",&(training->useMaxLowestLearntvsCutoff)); // = FALSE;
	getParamFileToggle("includePseudoInError", &(training->includePseudoInError));
	getParamFileToggle("realRehearsal", &(training->realRehearsal));
	getParamFileFloat("realRehearsalPercent", &(training->realRehearsalPercent));

	getParamFileToggle("noiseOnPseudoPatts", &(training->noiseOnPseudoPatts));
	getParamFileFloat("noiseRatioPseudoPatts", &(training->noiseRatioPseudoPatts));
	getParamFileFloat("learningRatioPseudoPatts", &(training->learningRatioPseudoPatts));

	getParamFileInt("maxPseudoPatts",&(training->maxPseudoPatts)); // = 40; //maximum number of patterns to store as Pseudo patterns
	getParamFileInt("maxSamplesForPseudoPatts",&(training->maxSamplesForPseudoPatts)); // = 1000; //How many sample to try to get the appropriate number of pseudo items
	getParamFileToggle("uniquePseudoPatts",&(training->uniquePseudoPatts)); // = TRUE;  //This prevents the pseudo items from being repeated.
	getParamFileToggle("pseudoRehearsalReal",&(training->pseudoRehearsalReal)); // = FALSE;  //This ensures that only learnt patterns are rehearsed

	getParamFileToggle("removeLearntFromPseudoPatts",
		&(training->removeLearntFromPseudoPatts)); // = FALSE; // true if you  want pure spurious patterns as the pseudo items.
	/*end 2*/
	/* was at 3 */
	getParamFileFloat("unlearningCutoffHigh",&(training->unlearningCutoffHigh));
	getParamFileFloat("unlearningCutoffLow",&(training->unlearningCutoffLow));
	getParamFileInt("unlearningCycleSize", &(training->unlearningCycleSize));
	getParamFileData("unlearningType", tempbuf);
	if (STR_EQ(tempbuf,"unlearningStandard")){
		training->unlearningType=unlearningStandard;
	}else{
		if (STR_EQ(tempbuf,"unlearningSpuriousRatio")){
			training->unlearningType=unlearningSpuriousRatio;
		}else{
			if (STR_EQ(tempbuf,"unlearningSpuriousMean")){
				training->unlearningType=unlearningSpuriousMean;
			}else{
				if (STR_EQ(tempbuf,"unlearningSpuriousOnly")){
					training->unlearningType=unlearningSpuriousOnly;
				}else{
					paramError("unlearningType");
				}
			}
		}
	}
	getParamFileToggle("unlearningSuperHeated", &(training->unlearningSuperHeated));
	getParamFileFloat("unlearningConst", &(training->unlearningConst));
	getParamFileToggle("uniqueUnlearningPatts",&(training->uniqueUnlearningPatts));
	getParamFileToggle("removeLearntFromUnlearningPatts",&(training->removeLearntFromUnlearningPatts));
	/*end 3*/


	getParamFileData("learningType",tempbuf);
	if (STR_EQ(tempbuf,"HEBBIAN")){ // Unlearning is generally a hebbian thing
		training->learningType = HEBBIAN;
		/*1*/
	} else {
		if (STR_EQ(tempbuf,"PSEUDODELTA")){
			training->learningType = PSEUDODELTA;
			/*2*/
		} else {
			if (STR_EQ(tempbuf,"STRICTDELTA")){
				training->learningType = STRICTDELTA;
			} else {
				if (STR_EQ(tempbuf,"UNLEARNING")){
					training->learningType = UNLEARNING;
					/*3*/
				} else {
					paramError("learningType");
				}
			}
		}
	}

	getParamFileInt("numCheck",&(training->numCheck)); // = 1000;
	getParamFileInt("maxSpuriousPatts",&(training->maxSpuriousPatts)); // = 1000;





	getParamFileFloat("errorCriteria",&(training->errorCriteria)); // = 0.001;

	getParamFileFloat("errorTail",&(training->errorTail)); // = 0.5;

	getParamFileData("errorCalcType",tempbuf);
	if(STR_EQ(tempbuf,"DELTAERROR")){
		training->errorCalcType=DELTAERROR;
	} else {
		if(STR_EQ(tempbuf,"SIGMOIDERROR")){
			training->errorCalcType=SIGMOIDERROR;
		} else {
			paramError("errorCalcType");
		}
	}// the type of error.


	getParamFileInt("initialNumberPatts",&(training->initialNumberPatts)); // =1; //16;
	getParamFileInt("step",&(training->step)); // = 1;
	getParamFileInt("numLearntPatts",&(training->numLearntPatts)); // = 5; //16;
	training->maxLearntPatts=MAX(training->maxLearntPatts,training->numLearntPatts);    



	getParamFileToggle("adjustLearningForPobability",&(training->adjustLearningForPobability)); // = FALSE; // this is the (Si - a)(Sj - a) rather than (Si)(Sj)

	getParamFileData("gaussianNoiseType",tempbuf); // = ABSOLUTE;
	if(STR_EQ("ABSOLUTE",tempbuf)){
		training->gaussianNoiseType = ABSOLUTE;
		getParamFileFloat("gaussianAbsoluteRange",&(training->gaussianAbsoluteRange)); // = 2;
	} else {
		if(STR_EQ("RELATIVE",tempbuf)){
			training->gaussianNoiseType = RELATIVE;
			getParamFileFloat("gaussianRelativeRange",&(training->gaussianRelativeRange)); // = 0.2; // noise for delta learning as related to mean weight 0.5); // = half the mean weight
		} else {
			paramError("gaussianNoiseType");
		}
	}

	getParamFileToggle("noiseOnInput",&(training->noiseOnInput)); // = 0; 
	/* The noise for delta learning.  By input I mean that we disturb the 
	unit input.  Thus units close to their decision surface will be affected by noise while units already strongly set will not increase their learning */
	getParamFileToggle("noiseOnOutput",&(training->noiseOnOutput)); // = FALSE; //not implemented yet

	getParamFileToggle("noiseHetroAssociative",&(training->noiseHetroAssociative)); // = 0; 
	/*Okay here is the deal with hetro noise.  It disrupts the "hetroNoiseLevel" level of units in the pattern.  
	It gives the network a single relaxation
	checks the relaxed state against the desired.  Ie did the network in one step get to the desired output.
	This did give a warning in the result when the network had 3 units that had 0 input.  This is bad for the ratio measure.  It also looks strange.  The reason for this is that the network, when given a small number of patterns, with asymmetric weight
	connections can have some units that will have 0 input and so have zero output.  If the unit is zero in all the patterns
	then it will have no errors.  This can be fixed in many ways.  
	One add noise on the unit input
	make the weights symmetric
	add enough patterns so that a unit is on in at least one pattern.
	*/

	if(training->noiseHetroAssociative){
		getParamFileFloat("hetroNoiseLevel",&(training->hetroNoiseLevel)); // = 0.1;
	}


	//-----------------------------------------------
	getParamFileFloat("generateProbOn",&(training->generateProbOn)); // = 0.5; //This is the proability of a unit being on in the generated patterns
	getParamFileToggle("generateActivationExactLevel",&(training->generateActivationExactLevel)); // = FALSE; // this force the units to be apriori correlated as there
	//is a exact number of units that are ON

	getParamFileToggle("reorderUnitsinPatts",&(training->reorderUnitsinPatts)); // 1;
	getParamFileToggle("calcSpuriousBasins",&(training->calcSpuriousBasins)); // 1;

	getParamFileToggle("calcLearntBasins",&(training->calcLearntBasins)); // 1;


	training->saveStabilityProfilePattern = (int *)calloc(training->maxLearntPatts,sizeof(int)); 

	training->learntOrder = (int *)calloc(training->maxLearntPatts,sizeof(int));   
	initSequence(training->learntOrder ,training->maxLearntPatts);   
	training->fixedOrder = (int *)calloc(MAX(training->maxLearntPatts,training->maxSpuriousPatts),sizeof(int));   
	initSequence(training->fixedOrder ,MAX(training->maxLearntPatts,training->maxSpuriousPatts));   


	if (getParamFileList("saveStabilityProfilePattern",training->saveStabilityProfilePattern)>0){
		training->saveStabilityProfilePatternCheck=1;
	}else{
		training->saveStabilityProfilePatternCheck=0;
	};// -1;
//printf(" list ---- %d %d %d %d\n",training->saveStabilityProfilePattern[1],training->saveStabilityProfilePattern[2],training->saveStabilityProfilePattern[3],training->saveStabilityProfilePattern[4]);
	getParamFileInt("numSpuriousPatsProfiles",&(training->numSpuriousPatsProfiles));

	getParamFileToggle("displayTestedPatterns",&(training->displayTestedPatterns)); // 1;
	getParamFileToggle("displayLearntPattUnits",&(training->displayLearntPattUnits)); // 1;
	getParamFileToggle("displayLearntPattInput",&(training->displayLearntPattInput)); // 0;
	getParamFileToggle("displayLearntPattNetInput",&(training->displayLearntPattNetInput)); // 0;
	getParamFileToggle("displayLearntPattSummary",&(training->displayLearntPattSummary)); // TRUE;

	getParamFileToggle("displaySpuriousPattUnits",&(training->displaySpuriousPattUnits)); // 0;
	getParamFileToggle("displaySpuriousPattInput",&(training->displaySpuriousPattInput)); // 0;
	getParamFileToggle("displaySpuriousPattNetInput",&(training->displaySpuriousPattNetInput)); // 0;
	getParamFileToggle("displaySpuriousPattSummary",&(training->displaySpuriousPattSummary)); // TRUE;


	getParamFileToggle("displaySpuriousPatts",&(training->displaySpuriousPatts)); // FALSE;
	getParamFileToggle("displaySpuriousPatternDetails",&(training->displaySpuriousPatternDetails)); // TRUE;


	return 1;
}
