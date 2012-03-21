#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "general.h"
#include "net.h"
#include "training.h"
#include "patternFileIO.h"

/********************************************************************************************************

*/
void saveWeights(netinfo * net){
  int to,from;
  FILE * weightFile;
  char weightFilename[200];

  weightFile = NULL;
  while(weightFile == NULL){
    system("ls *.weights");
    printf("Please enter weight file: ");
    fflush(stdout);
    scanf("%s\n",weightFilename);
    weightFile = fopen(weightFilename,"w");
  }
  fprintf(weightFile,"%d //Num units\n",net->N);
  fprintf(weightFile,"%d //Bias Unit\n",net->bias);
  FOR(to,net->N){
    FOR(from,net->N+(net->bias)){
      fprintf(weightFile," % 5.2f",(net->weight[to][from]));
    }
    fprintf(weightFile,"\n");
  }
  fclose(weightFile);
}

/********************************************************************************************************

*/
void loadWeights(netinfo * net){
  int to,from,N,bias;
  char junk[255];
  FILE * weightFile;
  unsigned char choice;
  int minN;
  char weightFilename[200];

  weightFile = NULL;
  while(weightFile == NULL){
    system("ls *.weights");
    printf("Please enter weight file: ");
    fflush(stdout);
    scanf("%s\n",weightFilename);
    weightFile = fopen(weightFilename,"r");
  }
  fscanf(weightFile,"%d%[^\n]\n",&N,junk);
  minN = net->N;
  fscanf(weightFile,"%d%[^\n]\n",&bias,junk);
  if (N != net->N){
    perror("Number of items in file does not match size of network.\n");
    if (N < net->N){
      perror("default responce is to set the rest of the units as inactive");
      minN=N;
    }else{
      fprintf(stderr,"default responce is to use just the first %d units of the %d in the file",net->N,N);
    }
    perror("Type Q to exit, or hit enter to continue");
    choice = getchar();
    if ((choice == 'Q')||(choice=='q'))
      exit(-1);
  }
  FOR(to,minN){
    FOR(from,minN+(bias)){//Note that this is the bias read in form the file
      fscanf(weightFile," %lf",&(net->weight[to][from]));
    }
  }
  fclose(weightFile);
}



/********************************************************************************************************
This function saves a pattern passed to it.  It will write the file as # and . for easy human reading
It will not store the bias as that is alway 1 if it is on.
*/
void savePatt(netinfo * net, FILE * patternFile, pattern * pat,int * order){
  int i;
  if (order == NULL){
    FOR(i,net->N){
      if (net->OFF==pat->unit[i])
        fprintf(patternFile,".");
      else
        fprintf(patternFile,"#");
    }
  }else{
    FOR(i,net->N){
      if (net->OFF==pat->unit[order[i]])
        fprintf(patternFile,".");
      else
        fprintf(patternFile,"#");
    }
  }
  fprintf(patternFile,"\n");
}


/********************************************************************************************************

*/
void saveAllLearntPatterns(netinfo * net, traininginfo * train, int * order){
  int i,validFile;
  FILE * patternFile;
  char patFilename[200];

  validFile = FALSE;
  patternFile = NULL;
  while (!validFile){
    printf("\nPlease input pattern file name: ");
    fflush(stdout);
    scanf("%s",patFilename);
    patternFile = fopen(patFilename,"w");
    if (patternFile != NULL)
      validFile = TRUE;
  }
  fprintf(patternFile,"%d // N\n",net->N);
  fprintf(patternFile,"%d // num patterns saved\n",train->numLearntPatts);  
  for(i=0;i<train->numLearntPatts; i++){
    savePatt(net,patternFile, train->learntPatts+i,order);
  }
  fclose(patternFile); 
}


/********************************************************************************************************
\param pat is the pattern to read data from the file patternFile into
\param numSpare the number of units in the file that will not be needed

*/
void loadPatt(netinfo * net,  FILE * patternFile, pattern * pat, int numSpare){
  int i;
  char readChar;

  i=0;
  readChar=getc(patternFile);
  while(i<net->N){
    if ((readChar == '.')||(readChar == '0')){
      pat->unit[i++]=net->OFF;
    } else {
      if ((readChar == '#')||(readChar == '1')){
        pat->unit[i++]=net->ON;
		pat->sum++;
      }
    }
    if(readChar == '/'){ //Clean out any line following a
      while((readChar=getc(patternFile)!='\n')){
        if (feof(patternFile)){
          perror("File reading problem for pattern reading");
          exit(-1);
        }
      }
    }
    if (feof(patternFile)){
      perror("File reading problem for pattern reading");
      exit(-1);
    }
    readChar= getc(patternFile);
  }
  i=0;
  while(i<numSpare){
    readChar=getc(patternFile);
    i++;
  }
};


void loadAllLearntPatterns(netinfo * net, traininginfo * train){
  int i,validFile,numSpare,N,numPatts;
  FILE * patternFile;
  char * patternFilename;
  char junk[200];

  validFile = FALSE;
  patternFile = NULL;
  patternFilename = train->patternFilename;
  if (!strcmp(patternFilename,"none")){
    while (!validFile){
      system("ls *.txt");
      printf("\nPlease input pattern file name: ");
      fflush(stdout);
      scanf("%s",patternFilename);
      patternFile = fopen(patternFilename,"r");
      if (patternFile != NULL)
        validFile = TRUE;
    }
  } else {
    patternFile = fopen(patternFilename,"r");
    if (patternFile == NULL) {
      perror("cannot open pattern file");
      exit(-1);
    }
  }
  fscanf(patternFile,"%d%[^\n]\n",&N,junk);
  if(N<net->N){
    perror("Number of units in patterns smaller than network try decreasing net->N\n");
    exit(-1);
  }
  fscanf(patternFile,"%d%[^\n]\n",&(numPatts),junk);
  numSpare = net->N - N;
  if(numPatts < train->maxLearntPatts){
    perror("File has too few patterns to be useful");
    exit(-1);
  } else {
    if(numPatts > train->maxLearntPatts){
      fprintf(stderr,"File will be processesed, Some patterns will not be loaded\n");
    }
    
    for(i=0;i<train->maxLearntPatts; i++){
      loadPatt(net,patternFile, train->learntPatts+i, numSpare);
      train->learntPatts[i].num = i;
      train->learntPatts[i].closestLearntPattern =i;
    }
  }
  fclose(patternFile);
};





