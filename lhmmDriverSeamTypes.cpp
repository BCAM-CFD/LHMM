/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
// Created by Nicolas Moreno. 03/01/2021
//@ nmorenoc@bcamath.org   -  nicolas.morenoch@gmail.com

//Driver code to execute fully Lagrangian heterogenous multiscale simulations. Both scales are modelled using LAMMPS
// Macroscale: Modified SDPD (SPH-like) with accesible stress and velocity gradient tensor for each macro particle
// Microscale: Modified SDPD implementation including explicitly bulk and shear viscosity and proper scaling for 2 and 3 dim 
// The code excutes macroscale simulations, and every tsamp time steps micro scale simulations are triggered to retrieve
// the stress tensor. Macroscales create each microscale simulation providing the velocity gradient for each macro particle.
// Microscales simulations are conducted for tmicro time steps, and the computed stress tensor is sent to macro. The process
// repeats untill tmacro time steps are completed.
// Communicators and creation of instances for micro are handle it by LAMMPS directly. Here we just assign the chunk of procs.
// to use for micro.
//
// Syntax: mpirun -np P lhmmDriver Nmac Nmic in.lammps.macro in.lammps.macro tmacro tsamp tmicro
//         P = # of total procs to run the driver program on
//         Nmac = # of processors used to solve macroscales
//         Nmic = # of processors used to solve microscales 
//         in.lammps.macro = LAMMPS input script
//         in.lammps.micro = LAMMPS input script to create microscale simulations 
//         tmacro = # of time steps to run macro scales
//         tsamp  = # of time steps to run before computing microscales  (this is, microscales are created every tsamp)
//         tmicro = # of time steps to run microscales 
// See README for compilation instructions

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "lammps.h"         // these are LAMMPS include files
#include "input.h"
#include "atom.h"
#include "library.h"

using namespace LAMMPS_NS;


void runMicros(LAMMPS *lmp[], int ninstance, int micpis, int shift, bool debug, MPI_Comm comm_micro, double gradv[][9], double stfluid[][6], double stmic[][6], int natoms, int indFluid[], int vfrequp);
void startMicros(char *infile, LAMMPS *lmp[], MPI_Comm comm_micro, int ninstance, int nFluidAtoms, float epsi,int &micpis,int &shift, int indFluid[]);
void runMacro (LAMMPS *lmp, int natoms, int tsamp, double gradv[][9], double stfluid[][6], double stmic[][6], int ct);
void sendStress();
void sendGradV(double gradv[]);
void retrieveStress();
void retrieveGradV(double gradv[]);


int main(int narg, char **arg)
{
  // setup MPI and various communicators
  // driver runs on all procs in MPI_COMM_WORLD
  // comm_lammps only has 1st P procs (could be all or any subset)

  //MPI_Init(&narg,&arg);
  MPI_Init(NULL, NULL);

  if (narg != 9) {
    printf("Syntax: lhmmDriver Nmac Nmic in.lammps.macro in.lammps.macro tmacro tsamp tmicro\n");
    exit(1);
  }

  int me,nprocs; 
  MPI_Comm_rank(MPI_COMM_WORLD,&me);  //world rank
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs); // world procs.

///Read parameters to run driver
  bool debug = true;  // by default always print controls

  int nprocs_macro = atoi(arg[1]);  //number of procs for macro scales
  int ninstance = atoi(arg[2]);   //for now is the number of instances to run for micro scales 
  //but for fully resolved should be (nproc-nproc_macro)/nparticles_macro
  char *infileMacro = arg[3]; 
  char *infileMicro = arg[4];
  int tmacro = atoi(arg[5]);
  int tsamp  = atoi(arg[6]);
  int tmicro = atoi(arg[7]);
  float epsi   = atof(arg[8]);
  int istest = 0; //If running driver then default test flag is set to zero.
  
///Pending to code error catching for inproper set of initial procs.
  if ((nprocs_macro) > (nprocs)-1) {
    if (me == 0)
      printf("ERROR: At least one proc must used for microscales\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

/// Creating group of procs for microscopic simulations
  MPI_Group world_group_id;  // list of groups in world
  MPI_Comm  micro_comm_id;   // communicator for micro
  MPI_Group micro_group_id;  // Created id for group
//  Get a group identifier for MPI_COMM_WORLD.
  MPI_Comm_group ( MPI_COMM_WORLD, &world_group_id );  // Adding list of groups

  // create one communicator per instancem each with P/N procs
  int instance;
  int micro_flag;
  if (me < nprocs_macro) micro_flag = 0;  //all ranks lower than procs in macro have the flag 0
  else micro_flag = 1;  //larger rank are used for micro


//  MPI_Comm comm_micro;  // creating sub comm for micro only
//  MPI_Comm_split(MPI_COMM_WORLD, micro_flag,0,&comm_micro); //splitted based on micro flag
//  printf("global rank %d \n", me);
  
  //int me_micro,nprocs_micro; 
  //MPI_Comm_rank(comm_micro,&me_micro); //Ranks of proc. for micro group
  //MPI_Comm_size(comm_micro,&nprocs_micro); // number of procs in micro group.

//  int me_micro,nprocs_micro; 
//  MPI_Comm_rank(comm_micro,&me_micro);
//  MPI_Comm_size(comm_micro,&nprocs_micro);

  int sample = 0;
  int T = 0; // total number of macro time steps run upto now
  int  t;
  LAMMPS *lmp = NULL;
  int natomsTotal = 0;
  int natoms =0;
  int nFluidAtoms = 0;


 
  /////////////////////////////here starts the macro scales
  if (micro_flag == 0){
      char str1[32],str2[32],str3[32], str4[32];

  char **lmparg = new char*[8];
  lmparg[0] = NULL;                 // required placeholder for program name
  //lmparg[1] = (char *) "-screen";
  //sprintf(str1,"screenMacro");
  //lmparg[2] = str1;
  lmparg[1] = (char *) "-log";
  sprintf(str2,"logMacro.lammps");
  lmparg[2] = str2;
  lmparg[3] = (char *) "-var";
  lmparg[4] = (char *) "t";
  sprintf(str3,"%d",tsamp);
  lmparg[5] = str3;

  lmparg[6] = (char *) "-var";
  lmparg[7] = (char *) "epsmm2";
  sprintf(str4,"%g",epsi);
  lmparg[8] = str4;
   
   lmp = new LAMMPS(9,lmparg,MPI_COMM_SELF);  //first arg is the number of argument passed from command line to lammsp 
   printf("MacroInstace %d\n", me);
    lammps_file(lmp,infileMacro);

    natoms = static_cast<int> (lmp->atom->natoms);
    printf("atoms in whole macro domain are %d\n", natoms);

  }

  //communicate info to all proc and create global arrays
  MPI_Bcast(&natoms, 1, MPI_INT, 0, MPI_COMM_WORLD);//sendGradV(gradv);

  MPI_Barrier(MPI_COMM_WORLD);

  double gradv[natoms][9] = {};
  double stfluid[natoms][6] = {};
  double stmic[natoms][6] = {};
  int types[natoms] = {}; 
  int indA[natoms] = {} ; //extracting macro-particle index to make easy handling
  int indFluid[nFluidAtoms] = {}; // array with index of fluid particles

  // populate index of fluid particles array to be available from all procs
  if (micro_flag == 0){
  	int *type = (int *) lammps_extract_atom(lmp,(char *) "type");
    int *inA = (int *) lammps_extract_atom(lmp,(char *) "id");
    for (int ii = 0; ii < natoms; ii++) { 
    	types[ii] = type[ii]; 
    	indA[ii] = inA[ii];
    	if (type[ii]==1) { 
    		indFluid[nFluidAtoms]=inA[ii]-1;  //value at each nFluid is the atom index only for type 1
    	 	nFluidAtoms+=1; 
    	}
    }
  	printf("atoms in fluid macro are %d\n", nFluidAtoms);
  	nFluidAtoms = static_cast<int> (nFluidAtoms);
  }

  MPI_Bcast(&types, natoms, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&indA, natoms, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
  MPI_Bcast(&nFluidAtoms, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Barrier(MPI_COMM_WORLD);

//need to be sure all the procs have nFluidAtoms
  MPI_Bcast(&indFluid, nFluidAtoms, MPI_DOUBLE, 0, MPI_COMM_WORLD);//   
  MPI_Barrier(MPI_COMM_WORLD);

  if (micro_flag==0) printf("Done Bcasting \n");


  int micpis;
  int shift;
  LAMMPS *lmpMic[nFluidAtoms] = {};
  MPI_Comm comm_micro;  // creating sub comm for micro only
  MPI_Comm_split(MPI_COMM_WORLD, micro_flag,0,&comm_micro); //splitted based on micro flag
  if (micro_flag==1) startMicros(infileMicro, lmpMic, comm_micro, ninstance, nFluidAtoms,epsi,micpis,shift,indFluid);
 // printf("micpis %d and shift %d \n", micpis, shift);
  MPI_Barrier(MPI_COMM_WORLD);

 if (micro_flag==0) printf("Done starting micros \n");

  for (int tt=0; tt<tmacro/tsamp; tt++){
        if (micro_flag==0) printf("Current time %d\n", T);
        sample = 0;

        if (micro_flag==0) runMacro(lmp, natoms, tsamp, gradv, stfluid, stmic,tt);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&gradv, natoms*9, MPI_DOUBLE, 0, MPI_COMM_WORLD);//sendGradV(gradv);
        //MPI_Bcast(&types, natoms, MPI_DOUBLE, 0, MPI_COMM_WORLD);//sendGradV(gradv);      
        MPI_Barrier(MPI_COMM_WORLD);

        if(me==1) printf("Gradv for microscales %g\n", gradv[3][3]);
         T+=tsamp;
         if (micro_flag==0) printf("Current time %d\n", T);

         if(micro_flag==1) runMicros(lmpMic, ninstance, micpis, shift,debug, comm_micro,gradv, stfluid, stmic,natoms,indFluid, tmicro);

        MPI_Barrier(MPI_COMM_WORLD);
        //MPI_Allreduce(temps,alltemps,ninstance,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        MPI_Bcast(&stmic, natoms*6, MPI_DOUBLE, 1, MPI_COMM_WORLD);      
        //MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&stfluid, natoms*6, MPI_DOUBLE, 1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if(me==0) printf("stfluid for macroscales %g %g\n", stfluid[3][3], stfluid[20][3]);
        
    }
    if (micro_flag==0) lmp->input->one("run 1");
    printf("gradv 2 is %g\n", gradv[3][0]);
    printf("stmic 2 is %g\n", stmic[3][1]);
     delete lmp;
  
  //if (micro_flag ==1){
    //  printf("here goes the micro");
          //MPI_Barrier(MPI_COMM_WORLD);
            //MPI_Finalize();

    //printf("Rank of receiver %d\n", me);
    //MPI_Barrier(MPI_COMM_WORLD);

   // MPI_Bcast(&gradv, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);//retrieveGradV(gradv);
   // if(me==1) printf("Gradv for microscales %g\n", gradv[3]);

  //    runMicros(infileMicro, tmicro,  ninstance, debug, comm_micro);
  //    sendStress();
  //} 

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm_free(&comm_micro);

  MPI_Finalize();
}


//////////////////////////////////////// MACRO /////////////////////////////////////////////////////////

void runMacro (LAMMPS *lmp, int natoms, int tsamp, double gradv[][9],double stfluid[][6], double stmic[][6], int tt){
    int *inA = (int *) lammps_extract_atom(lmp,(char *) "id");
    int ii,jj;
    //Retrieving the pointers for stress for fluid and microstructure to set it as the current value 
    double **sf = (double **) lammps_extract_atom(lmp,(char *) "stfluid");
    double **sm = (double **) lammps_extract_atom(lmp,(char *) "stmic");
    //k index run over lammps unsorted arrays sf,sm and gv. Whereas ii and jj run over sorted index
    //consistent with the particle ID
    for (int kk = 0; kk < natoms; kk++) {
      	ii = inA[kk]-1;
        sm[kk][0] = stmic[ii][0];
        sm[kk][1] = stmic[ii][1];
        sm[kk][2] = stmic[ii][2];
        sm[kk][3] = stmic[ii][3];
        sm[kk][4] = stmic[ii][4];
        sm[kk][5] = stmic[ii][5];
      
        sf[kk][0] = stfluid[ii][0];
        sf[kk][1] = stfluid[ii][1];
        sf[kk][2] = stfluid[ii][2];
        sf[kk][3] = stfluid[ii][3];
        sf[kk][4] = stfluid[ii][4];
        sf[kk][5] = stfluid[ii][5];
    }

    //printf("stfluid for an atom: %g %g\n",sm[50][0],sm[50][1]);

    //delete type;
    char str1[32];
    sprintf(str1,"run %d pre no post no",tsamp);
    lammps_command(lmp, str1);

    double **gV = (double **) lammps_extract_atom(lmp,(char *) "gradv");
 //   type = (int *) lammps_extract_atom(lmp,(char *) "type");
    inA = (int *) lammps_extract_atom(lmp,(char *) "id");

    //double **pos = (double **) lammps_extract_atom(lmp,(char *) "x");
    //double **id = (double **) lammps_extract_atom(lmp,(char *) "id");

   // printf("gv %g\n", gV[100][4]);
    ///this is dsitributed per proc so the values obtained for particle i are only for the current proc
    //int *ind = (int *) lammps_extract_atom(lmp,(char *) "id");

    ////////// TO VERIFY IF AFTER MACRO STEP THE INDEX OF THE ARRAYS ARE CONSISTENT AND THE TYPES IS STILL CONSISTENT. 

    double trace;
    double dim = 2;   

    for (int ll = 0; ll < natoms; ll++) {
      	  jj = inA[ll]-1;

	      trace = (gV[ll][0]+gV[ll][1]+gV[ll][2])/dim;
	      gradv[jj][0] = gV[ll][0]-trace;
	      gradv[jj][1] = gV[ll][1]-trace;
	      
	      gradv[jj][3] = gV[ll][3];
	      gradv[jj][6] = gV[ll][6];

	      if (dim==3){ 
	      gradv[jj][2] = gV[ll][2]-trace;
	      gradv[jj][4] = gV[ll][4];
	      gradv[jj][5] = gV[ll][5];
	      
	      gradv[jj][7] = gV[ll][7];
	      gradv[jj][8] = gV[ll][8];
      }
    }

    return;
}

/////////////////   MICROSCALES ////////////////////////////////////////////

void startMicros(char *infile, LAMMPS *lmp[], MPI_Comm comm_micro, int ninstance, int nFluidAtoms, float epsi, int &micpis,int &shift, int indFluid[]){
  int istest = 0;
  int instance;
  int rank_micro,nprocs_micro; 
  MPI_Comm_rank(comm_micro,&rank_micro);
  MPI_Comm_size(comm_micro,&nprocs_micro);
  
 // if (debug) printf("Proc. micro %d, rank in micro %d \n", nprocs_micro, me_micro);

  //printf("procs %d, rankMicro %d \n", nprocs_micro, me_micro);

  instance = (rank_micro)*(ninstance) / (nprocs_micro);  // the plus 1 correspond to the instance of the macro scale 
  //printf("instance %d, me %d \n", instance, me);

  //else instance = (me-nprocs_macro)*(ninstance) / (nprocs-nprocs_macro);  // the plus 1 correspond to the instance of the macro scale 

  MPI_Comm comm_lammps;
  
  MPI_Comm_split(comm_micro,instance,1,&comm_lammps);
  //  MPI_Comm_split(MPI_COMM_WORLD,instance,0,&comm_lammps);

  int rank_lammps;
  MPI_Comm_rank(comm_lammps,&rank_lammps);

  micpis = nFluidAtoms/ninstance; //# of microscales subsystems per instance. 
  int leftmics = nFluidAtoms%ninstance; //leftover mics to distribute on the instances 
  for(int ll=0; ll<leftmics;ll++){
    if (rank_micro==ll) micpis+=1; //Distributing leftover microscales over lammps instances
  }

  shift = 0; //to shift the array when placing the data on the gradv and stmic vectors 
  int micindex=0;
  if(rank_micro>=leftmics) shift = leftmics;  

  // open N instances of LAMMPS

  //double *ptr = NULL; 
  ///Each micro proc iterates over the number of micro simulations it has to do
  for(int mm=0; mm<micpis;mm++){
      micindex = shift+mm+rank_micro*micpis; //index of the microsytem
      //printf("rank_micro %d with micindex %d, with total %d procs\n", rank_micro, micindex, micpis);
     //if (types[micindex]==1){
      char str1[32],str2[32],str3[64];
      char **lmparg = new char*[5];
      lmparg[0] = NULL;                 // required placeholder for program name
      lmparg[1] = (char *) "-screen";
      sprintf(str1,"none");
      //sprintf(str1,"mics/screen.%d",micindex);
      lmparg[2] = str1;
      lmparg[3] = (char *) "-log";
      sprintf(str2,"mics/log.lammps.%d",indFluid[micindex]);
      lmparg[4] = str2;
      
       lmp[micindex] = NULL;
       lmp[micindex] = new LAMMPS(5,lmparg,comm_lammps);  //first arg is the number of argument passed from command line to lammsp 
       //lammps_file(lmp[micindex],infile);

       sprintf(str3,"variable imic equal %d",indFluid[micindex]) ;   //index to print files per micro intance for debugging.
       char *strtwo = (char *) str3;
       lammps_commands_string(lmp[micindex],strtwo);
       //printf("done at setimic %d", rank_micro);


       sprintf(str3,"variable tmac equal %d",rank_micro) ;
       strtwo = (char *) str3;
       lammps_commands_string(lmp[micindex],strtwo);

       sprintf(str3,"variable test equal %d",istest) ;
       strtwo = (char *) str3;
       lammps_commands_string(lmp[micindex],strtwo);
         // run input script thru all instances of LAMMPS

       sprintf(str3,"variable epsmm2 equal %g",epsi) ;
       strtwo = (char *) str3;
       lammps_commands_string(lmp[micindex],strtwo);

       char str4[64];
       
       //sprintf(str3,"variable t equal %d",tmicro) ;
       //strtwo = (char *) str3;
       //lammps_commands_string(lmp,strtwo);

       sprintf(str4,"variable gdvxx equal %g",0.0) ;
       char *strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvyy equal %g",0.0) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvzz equal %g",0.0) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvxy equal %g",0.000001) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvxz equal %g",0.0) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvyz equal %g",0.0) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvyx equal %g",0.000001) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvzx equal %g",0.0) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvzy equal %g",0.0) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

      lammps_file(lmp[micindex],infile);
      lammps_command(lmp[micindex], "run 10");

      delete [] lmparg;
      //}
    }

  //    printf("in start micpis %d and shift %d \n", micpis, shift);

}


void runMicros(LAMMPS *lmp[], int ninstance, int micpis, int shift, bool debug, MPI_Comm comm_micro, double gradv[][9], double stfluid[][6], double stmic[][6], int natoms, int indFluid[], int vfrequp){
  int rank_micro,nprocs_micro; 
  MPI_Comm_rank(comm_micro,&rank_micro);
  MPI_Comm_size(comm_micro,&nprocs_micro);
  //double *temps = new double[ninstance];
  //double *alltemps = new double[ninstance];
  double sfl[natoms][6] = {};
  double smi[natoms][6] = {};

  double pressF;
  double pressM;

  double etascale = 1; //a value of 1e-4 corresponds to eta microscopi of 10
  int dim = 2;

  double *ptrf = NULL;
  double *ptrm = NULL;
  
  // open N instances of LAMMPS
  int micindex=0; //index over fluid atoms
  int fluidIndex=0; //index or identifier of fluid particle to allocate in global arrays of gdv, sf, and sm
  ///Each micro proc iterates over the number of micro simulations it has to do
  for(int mm=0; mm<micpis;mm++){
      micindex = shift+mm+rank_micro*micpis; //index of the microsytem
      fluidIndex = indFluid[micindex];
     // printf("rank_micro %d with micindex %d, with total %d procs\n", rank_micro, micindex, micpis);
      //lammps_command(lmp[micindex], "run 10");
      char str4[64],str3[32];

      sprintf(str3,"log mics/log.lammps.%d",fluidIndex);
      char *strtwo = (char *) str3;
      lammps_commands_string(lmp[micindex],strtwo);
       
       //sprintf(str3,"variable t equal %d",tmicro) ;
       //strtwo = (char *) str3;
       //lammps_commands_string(lmp,strtwo);

       sprintf(str4,"variable gdvxx equal %g",gradv[fluidIndex][0]) ;
       char *strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvyy equal %g",gradv[fluidIndex][1]) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvzz equal %g",gradv[fluidIndex][2]) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvxy equal %g",gradv[fluidIndex][3]) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvxz equal %g",gradv[fluidIndex][4]) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvyz equal %g",gradv[fluidIndex][5]) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvyx equal %g",gradv[fluidIndex][6]) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable gdvzx equal %g",gradv[fluidIndex][7]) ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       /// It is needed to redefine VX and VY in order to be recomputed with the new gradients by lammps. otherwise the value will be fix all  the simulation
       sprintf(str4,"variable VX atom x*${gdvxx}+y*${gdvxy}") ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

       sprintf(str4,"variable VY atom y*${gdvyy}+x*${gdvyx}") ;
       strfour = (char *) str4;
       lammps_commands_string(lmp[micindex],strfour);

      // run input script thru all instances of LAMMPS

       char str1[64];

       //sprintf(str1,"set group cons vx v_VX vy v_VY");
       //lammps_command(lmp[micindex], str1);

       sprintf(str1,"run %d pre no post no every 10 \"set group cons vx v_VX vy v_VY\" ",vfrequp);  ///gonna use the velocity-frequency update to set how many steps run in micro before exchangin data with macr
       lammps_command(lmp[micindex], str1);

      //delete strtwo;
      //delete strfour;
     // int ierr = lammps_has_error(lmp);
     // if ( ierr != 0 ){
     //    printf("Microscales - Fatal error!\n");
     //     exit ( 1 );
     ///  }
      //
      ptrf = (double *) lammps_extract_fix(lmp[micindex],(char *) "piF",0,1,0,0);
      pressF = *ptrf;
      ptrf = (double *) lammps_extract_fix(lmp[micindex],(char *) "piF",0,1,1,0);
      sfl[fluidIndex][0] =  (*ptrf+pressF)*etascale;       //////////////////////Temporarily set no contribution of pressure
      ptrf = (double *) lammps_extract_fix(lmp[micindex],(char *) "piF",0,1,2,0);
      sfl[fluidIndex][1] =  (*ptrf+pressF)*etascale;
      ptrf = (double *) lammps_extract_fix(lmp[micindex],(char *) "piF",0,1,4,0);
      sfl[fluidIndex][3] =  *ptrf*etascale;

      ptrm = (double *) lammps_extract_fix(lmp[micindex],(char *) "piM",0,1,0,0);
      pressM = *ptrm;
      ptrm = (double *) lammps_extract_fix(lmp[micindex],(char *) "piM",0,1,1,0);
      smi[fluidIndex][0] =  (*ptrm+pressM)*etascale;       //////////////////////Temporarily set no contribution of pressure
      ptrm = (double *) lammps_extract_fix(lmp[micindex],(char *) "piM",0,1,2,0);
      smi[fluidIndex][1] =  (*ptrm+pressM)*etascale;
      ptrm = (double *) lammps_extract_fix(lmp[micindex],(char *) "piM",0,1,4,0);
      smi[fluidIndex][3] =  *ptrm*etascale;


      if (dim==3){ 
        ptrf = (double *) lammps_extract_fix(lmp[micindex],(char *) "piF",0,1,3,0);
         sfl[fluidIndex][2] =  (*ptrf+pressF)*etascale;  
        ptrf = (double *) lammps_extract_fix(lmp[micindex],(char *) "piF",0,1,5,0);
         sfl[fluidIndex][4] =  *ptrf*etascale;
        ptrf = (double *) lammps_extract_fix(lmp[micindex],(char *) "piF",0,1,6,0);
         sfl[fluidIndex][5] =  *ptrf*etascale;
       
        ptrm = (double *) lammps_extract_fix(lmp[micindex],(char *) "piM",0,1,3,0);
         smi[fluidIndex][2] =  (*ptrm+pressM)*etascale;  
        ptrm = (double *) lammps_extract_fix(lmp[micindex],(char *) "piM",0,1,5,0);
         smi[fluidIndex][4] =  *ptrm*etascale;
        ptrm = (double *) lammps_extract_fix(lmp[micindex],(char *) "piM",0,1,6,0);
         smi[fluidIndex][5] =  *ptrm*etascale;
       }
       //lammps_free((void *)ptr);
       // lammps_free((void *)ptr2);


      //printf("Pressure %g\n", stfluid[micindex][3]);
      
      //stmic[micindex][0] = 0.;
      //stmic[micindex][1] = 0.;
      //stmic[micindex][2] = 0.;
      //stmic[micindex][3] = 0.;
      //stmic[micindex][4] = 0.;
      //stmic[micindex][5] = 0.;
     
      //grav[remai+i+ran*atpran[ran]]+=1 //this is to set the values at location of the microsystem.
    }
   // delete ptrf;
    //delete ptrm;

   //
      
  //for (int i = 0; i < ninstance; i++) temps[i] = 0.0;

  
  //if (me_lammps == 0) temps[instance] = finaltemp;
  //printf("The final temp for %d is %g", me, finaltemp);
  MPI_Barrier(comm_micro);
  MPI_Allreduce(sfl,stfluid,natoms*6,MPI_DOUBLE,MPI_SUM,comm_micro);
  MPI_Allreduce(smi,stmic,natoms*6,MPI_DOUBLE,MPI_SUM,comm_micro);

  //delete lmp;
  //if(me_lammps==0){
   // for (int i = 0; i < ninstance; i++)
  //    printf("Instance %d, final temp = %g\n",i+1,alltemps[i]);
  //}

  //delete [] sfl;
  //delete [] alltemps;

  // delete LAMMPS instances

 // delete lmp;
  // close down MPI

  //MPI_Comm_free(&comm_lammps);
  return;
}
