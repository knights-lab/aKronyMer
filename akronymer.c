#define VER "v1.00"
#define USAGE "usage: aKronyMer inseqs.lin.fna output [K] [Q queries.lin.fna [MIN2]] [HEUR[0-9]] [ANI] [CHANCE] [GC] [ADJ] [GLOBAL/DIRECT/LOCAL] [TREE/SETCLUSTER] [RC] [NOCUT]"
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <float.h>
#include <math.h>
#ifndef __SSE4_2__
	#define _mm_popcnt_u64 __builtin_popcountll
#endif
//                           A   C       G                         T U
const uint8_t CONV[32]  = {4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4,4,4,4,4,3,3,4,4,4,4,4,4,4,4,4,4};
//                         0 0 0 0 0 Z Y X W V U T S R Q P O N M L K J I H G F E D C B A 0
const uint8_t RCONV[32] = {4,3,4,2,4,4,4,1,4,4,4,4,4,4,4,4,4,4,4,4,0,0,4,4,4,4,4,4,4,4,4,4};
__m128i MSK[16];

double K_recip, rspace;
uint64_t Kspace_denom;

void * malloc_a(size_t algn, size_t size, void **oldPtr) {
	uintptr_t mask = ~(uintptr_t)(algn - 1);
	*oldPtr = malloc(size+algn-1);
	return (void *)(((uintptr_t)*oldPtr+algn-1) & mask);
}

static inline double calc_dist(uint32_t pop1, uint32_t pop2, double gc1, double gc2, int its, int unn, 
int doChance, int doGC, int ani, int direct, int adj, int global, int local) {
	uint32_t l = pop1, h = pop2, t;
	if (l > h) t = l, l= h, h = t;
	//double tF, hGC = gc1, lGC = gc2; 
	//if (hGC < lGC) tF = hGC, hGC = lGC, lGC = tF;
	
	// Distance calculation
	double chance = 0., forceps = 0.;
	if (doChance) {
		if (doGC) {
			double gc1x = 2.0*(gc1-.5), gc2x = 2.0*(gc2-.5), tF;
			double gc1gc2 = gc1x*gc2x;
			double convergent = gc1gc2 > 0.;
			double p1x = rspace*pop1, p2x = rspace*pop2;

			double absWorseGC = gc1x < 0 ? -gc1x : gc1x, absBetterGC = gc2x < 0 ? -gc2x: gc2x; 
			double wGCpop = p1x, bGCpop = p2x;
			if (absWorseGC < absBetterGC) tF = absWorseGC, absWorseGC = absBetterGC, absBetterGC=tF,
				wGCpop = p2x, bGCpop = p1x;
			
			double naiveI = p1x + p2x - (1.-(1.-p1x)*(1.-p2x));
			double logNaiveI = naiveI > 0 ? log(naiveI) : log(rspace);

			double logInt = -0.059930023639 + 8.80552018132*gc1gc2 + 0.982605251473*logNaiveI + 
				-10.698650968033*absWorseGC*bGCpop*gc1gc2 + -10.096185620097*convergent*absBetterGC*gc1gc2 + 
				-0.6996572809*absBetterGC*gc1gc2*logNaiveI + 41.38264645886*absBetterGC*absWorseGC*naiveI + 
				0.779298577831*absBetterGC*bGCpop*logNaiveI + -4.26363078487*convergent*absWorseGC*wGCpop + 
				0.307017388822*absBetterGC*absWorseGC*logNaiveI + 0.195881822435*convergent + -31.798178947318*
				absBetterGC*wGCpop*naiveI + -13.147853260478*absWorseGC*wGCpop*gc1gc2 + 1.922706924807*
				convergent*absBetterGC;
			chance = exp(logInt)*Kspace_denom;
		}
		else chance = rspace * (double)l * (double)h;
	}
	//double adj1 = pow(.1/(1.0 - betterGC),1.33)*21.9796; // convergence boost
		
	//printf("Chance multiplier for pop1 %u, pop2 %u, hGC %f, lGC %f, its %d, unn %d = %.0f [orig %.0f]\n",
	//	pop1,pop2,gc1,gc2,its,unn, chance,rspace*pop1*pop2);
	
		chance = chance > its ? its : chance;
	//printf("%u\t%u\t%f\t%f\t%d\t%d\t%f\t%f\t%f\t%f\n",
	//	pop1,pop2,hGC,lGC,its,unn, chance,rspace*pop1*pop2,betterGC,worstGC);
	//printf("%f\t%f\t%f\t%f\t%f\n",gc1,gc2,1-rspace*pop1, 1-rspace*pop2,1-rspace*unn);
	//printf("%f\t%f\t%g\t%g\t%g\n",gc1,gc2,rspace*pop1, rspace*pop2,rspace*its);
	double denom;
	if (ani) {
		// adjust numerator and denominator here
		double fudge = local ? (double)l/(double)h : 1.0;
		denom = unn * fudge;
		its -= chance;
		denom -= chance * fudge;
	}
	else if (direct) denom = h + l - its;
	else denom = global ? h : l; // global, glocal
	denom = denom > 1. ? denom : 1.;
	
	double sim = (double)its / denom, dist = 1.0 - sim;
	if (ani) { // LBA fix 
		if (!direct) {
			dist = 1. - (1. + K_recip*log(2*sim/(1. + sim)));
			if (adj) dist = dist < .75 ? -0.75*log(1.-(4./3)*dist) : 3,
				dist = dist > 3 ? 3 : dist;
			else dist = dist > 1 ? 1 : dist;
		}
		else if (adj) dist = dist <= .9999999 ? -log(1-dist) : 16.118096;
	} 
	else if (adj) {
		double nu;
		if (direct) nu = (double)h*l/(h+l);
		else nu = global ? l : h;
		double rd = 1. - nu*rspace;
		dist = dist >= rd ? 1 : dist/rd;
		dist = dist <= .9999999 ? -log(1-dist) : 16.118096;
	}
	return dist < 0 ? 0 : dist;
}
void main(int argc, char *argv[]) {
	puts("This is aKronyMer " VER " by Gabe.");
	if (argc < 3) {puts(USAGE); exit(1);}
	FILE *fp = fopen(argv[1],"rb"), *f2 = 0;
	if (!fp) {fprintf(stderr,"ERROR: input file I/O\n"); exit(1);}
	FILE *of = fopen(argv[2],"wb");
	if (!of) {fprintf(stderr,"ERROR: output file I/O\n"); exit(1);}
	const uint32_t SEQLEN = INT32_MAX, HEADLEN = INT32_MAX; //1 << 24, 1 << 16;
	char *head = malloc(HEADLEN + 1), *headO = head,
		*seq = calloc(SEQLEN + 16,1), *seqO = seq;
	head[HEADLEN] = 0, seq[SEQLEN] = 0;
	uint32_t i = 0, fx, profSz = 10, maxL = 0, THREADS = omp_get_max_threads(); 
	printf("Using %u thread(s).\n",THREADS);
	uint64_t totL = 0, N;
	int noCut = argc >= 4 && !strcmp(argv[argc-1],"NOCUT"); 
	argc -= noCut;
	int doRC = argc >= 4 && !strcmp(argv[argc-1],"RC"); 
	argc -= doRC;
	int doTree = argc >= 4 && !strcmp(argv[argc-1],"TREE"); 
	argc -= doTree;
	int doSetClus = argc >= 4 && !strcmp(argv[argc-1],"SETCLUSTER"); 
	argc -= doSetClus;
	int global = argc >= 4 && !strcmp(argv[argc-1],"GLOBAL");
	argc -= global;
	int direct = argc >= 4 && !strcmp(argv[argc-1],"DIRECT");
	argc -= direct;
	int local = argc >= 4 && !strcmp(argv[argc-1],"LOCAL");
	argc -= local;
	int adj = argc >= 4 && !strcmp(argv[argc-1],"ADJ");
	argc -= adj;
	int doGC = argc >= 4 && !strcmp(argv[argc-1],"GC");
	argc -= doGC;
	int doChance = argc >= 4 && !strcmp(argv[argc-1],"CHANCE");
	argc -= doChance;
	int ani = argc >= 4 && !strcmp(argv[argc-1],"ANI");
	argc -= ani;
	char *hr = 0, hlv = 0;
	int heur = argc >= 4 && (hr=strstr(argv[argc-1],"HEUR"));
	argc -= heur;
	int doMin2 = argc >=4 && !strcmp(argv[argc-1],"MIN2");
	argc -= doMin2;
	if (argc > 5 && !strcmp(argv[argc-2],"Q")) {
		if (doTree || doSetClus) {
			puts("Sorry, multi-file only supported for distance matrix output");
			exit(2);
		}
		printf("Also using file 2: %s\n",argv[argc-1]);
		f2 = fopen(argv[argc-1],"rb"), argc -=2;
		if (!f2) {printf("Well that didn't work. I/O error\n"); exit(2);}
		
	}

	if (hr) hlv = atoi(hr+4);
	printf("Goal: output %s chance-%scorrected %sANI %s %s %s RC\n", adj ? "adjusted" : "raw", 
		doChance? "" : "un", ani? "" : "non-", global ? "global" : direct ? "direct" : 
		local? "local" : "glocal", doTree ? "tree" : doSetClus ? "set clusters" : "distance matrix", 
		doRC ? "with" : "without");
	if (heur) printf("WARNING: Using lv %u setcov heuristic!\n", hlv);
	char **HeadPack = malloc(profSz*sizeof(*HeadPack));
	uint32_t *Lens = malloc(profSz*sizeof(*Lens));
	if (!(head && seq && HeadPack && Lens)) 
		{fputs("OOM:Init\n",stderr); exit(3);}
	double wtime = omp_get_wtime();
	
	// Prepass -- evaluate K, check fasta, get num seqs
	while (head = fgets(head,HEADLEN,fp)) {
		char *h = strchr(head,'\n');
		if (!h || *head != '>') {
			head[1024] = 0;
			if (!h) fprintf(stderr, "ERROR: head %u no NL: '%s'\n",i,head);
			else fprintf(stderr,"ERROR: head %u no '>': '%s'\n",i,head); 
			exit(2);
		}
		HeadPack[i] = malloc(h - head);
		if (!HeadPack[i]) {fputs("OOM:HeadPack_i\n",stderr); exit(3);}
		*h = 0, memcpy(HeadPack[i],head+1,h-head);
		seq = fgets(seq,SEQLEN,fp);
		if (!seq) {fprintf(stderr,"ERROR: sequence ln %u\n",i); exit(2);}
		Lens[i] = strlen(seq); 
		Lens[i] -= seq[Lens[i]-1] == '\n';
		totL += Lens[i], maxL = Lens[i] > maxL ? Lens[i] : maxL;
		if (++i == profSz) {
			HeadPack = realloc(HeadPack,(profSz*=2)*sizeof(*HeadPack));
			Lens = realloc(Lens,profSz*sizeof(*Lens));
			if (!HeadPack || !Lens) {fputs("OOM:HeadPack\n",stderr); exit(3);}
		}
	}
	uint32_t sugK = (31-__builtin_clz((totL+totL)/(N=i)))/2u + 1; //clz(totL/(N=i)+maxL)
	printf("Avg. length: %lu, max = %u. Sugg. K = %u\n", totL/N, maxL, sugK);
	if (N < 2 && !f2) {
		fputs("Sorry, need > 1 sequence!\n",stderr); 
		if (!doTree && !doSetClus) {
			// get name of ref and make faux dm
			rewind(fp);
			head = headO;
			head = fgets(head,HEADLEN,fp);
			fprintf(of,"\t%s%s",head+1,head+1);
		}
		exit(1);
	}
	uint32_t K = argc > 3 ? atoi(argv[3]) : sugK, PSZ, K1, FPSZ, FPSZH;
	K = K < 16 ? K : 16, K = K > 4 ? K : 4, K1 = K - 1;
	Kspace_denom = (uint64_t)1 << (2*K);
	if (doRC) Kspace_denom = Kspace_denom/2 + ((K & 1) ? 0 : (1 << (K-1))); // palindrome support
	K_recip = 1. / K, rspace = 1./Kspace_denom; // GLOBALS
	PSZ = 2*K-3; FPSZ = 1 << (PSZ-3), FPSZH = FPSZ >> hlv; 
	FPSZH += !FPSZH;
	printf("Running with K = %u [H = %u]\n",K,FPSZH);
	HeadPack = realloc(HeadPack,N*sizeof(*HeadPack));
	Lens = realloc(Lens,N*sizeof(*Lens));
	uint32_t *Pops = malloc(N*sizeof(*Pops));
	double *GC = malloc(sizeof(*GC)*N), *ND = malloc(sizeof(*ND)*N);
	uint64_t **ProfPack = malloc(N*sizeof(*ProfPack));
	uint64_t *ProfDump = calloc((uint64_t)N*FPSZ,sizeof(*ProfDump));
	if (!ProfPack || !Pops || !ProfDump) {fputs("OOM:Dump\n",stderr); exit(3);}
	for (uint64_t j = 0; j < N; ++j) ProfPack[j] = ProfDump + j*FPSZ;
	i = fx = 0, head = headO, seq = seqO;
	rewind(fp);
	uint32_t maxPop = 0, whichMaxPop = 0;
	while (head = fgets(head,HEADLEN,fp)) {
		seq = fgets(seq,SEQLEN,fp);
		seq[Lens[i]-1] = 0;
		uint8_t *P = (uint8_t *)ProfPack[fx];
		uint32_t fp = 0, len = Lens[i];
		uint16_t re = (uint16_t)-1 << (16 - K);
		uint32_t totGC = 0, totN = 0;
		#pragma omp parallel
		{
			#pragma omp for schedule(dynamic,256) reduction(+:totGC,totN)
			for (uint32_t j = K1; j < len; ++j) {
				char *s = seq + j - K1;
				uint32_t w = 0, wrc = 0;
				uint32_t testLet = CONV[seq[j-K1] & 31];
				if (testLet == 1 || testLet == 2) ++totGC;
				else if (testLet > 3) ++totN;
				
				for (uint32_t k = 0; k < K; ++k) {
					uint32_t x = CONV[s[k] & 31];
					if (x > 3u) goto ENDR;
					w |= x << (k << 1);
				}
				if (doRC) {
					for (int k = K1; k >= 0; k--) {
						uint32_t x = RCONV[s[k] & 31];
						wrc |= x << ((K1-k) << 1);
					}
					//#pragma omp atomic
					//P[wrc >> 3] |= 1 << (wrc & 7); 
					if (wrc < w) w = wrc;
				}

				#pragma omp atomic
				P[w >> 3] |= 1 << (w & 7); 
				ENDR:NULL;
			}
			#pragma omp single
			for (uint32_t j = len-K1; j < len; ++j) {
				uint32_t testLet = CONV[seq[j-K1] & 31];
				if (testLet == 1 || testLet == 2)
					++totGC;
				else if (testLet > 3) ++totN;
			}
			uint64_t *Agg = ProfPack[fx];
			#pragma omp for reduction(+:fp)
			for (uint32_t j = 0; j < FPSZ; ++j)
				fp += _mm_popcnt_u64(Agg[j]);
		}
		double newD = Kspace_denom;
		double thisGC = (double)totGC / (double)(len-totN);
		GC[i] = doGC ? thisGC : 0.5;
		
		printf("[%u (%u)] L = %u, Density = %u [%f], Entropy = %f, GC = %f\n",i, fx, len, fp, 
			(double)fp/newD,(double)fp/len, thisGC);
		if (fp > maxPop) maxPop = fp, whichMaxPop = fx;
		if (!heur) Pops[fx++] = fp + !fp;
		else { // Do heuristic search (direct==global here)
			if (!fp) {free(HeadPack[i++]); continue;} 
			int fb = -1;
			uint64_t *Cur = ProfPack[fx];
			#pragma omp parallel for
			for (uint32_t j = 0; j < fx; ++j) if (fb == -1) { // false break
				uint64_t its = 0, *S = ProfPack[j];
				for (uint32_t k = 0; k < FPSZH; ++k) 
					if (S[k] != Cur[k]) {its = 1; break;}
				if (!its) fb = j;
			}
			if (fb == -1) Pops[fx] = fp, HeadPack[fx++] = HeadPack[i];
			else {
				char *Us = malloc(strlen(HeadPack[fb])+strlen(HeadPack[i])+24);
				sprintf(Us,"(%s:%.5f,%s:%.5f)",HeadPack[fb],0.f,HeadPack[i],0.f);
				free(HeadPack[i]), free(HeadPack[fb]);
				HeadPack[fb] = Us; // no fx inc
				memset(ProfPack[fx],0,1 << PSZ);
			}
		}
		++i;
	}
	N = fx;
	free(seqO); free(headO); free(Lens); fclose(fp);
	if (N < i) ProfDump = realloc(ProfDump,(uint64_t)N*(1 << PSZ)),
		ProfPack = realloc(ProfPack,N*sizeof(*ProfPack));
	printf("Parsed %u (%lu cls) reference sequences [%f]\n",i,N,omp_get_wtime()-wtime);
	
	wtime = omp_get_wtime();
	if (doSetClus) {
		// Do greedy set coverage. 
		uint32_t *Set = calloc(N,sizeof(*Set));
		uint64_t *SetFP = calloc(FPSZ,sizeof(*SetFP));
		uint8_t *Mask = calloc(N,sizeof(*Mask));
		
		Set[0] = whichMaxPop; Mask[whichMaxPop] = 1;
		for (uint32_t i = 0; i < FPSZ; ++i) SetFP[i] = ProfPack[whichMaxPop][i];
		uint32_t numSet = 1, setPop = maxPop, setInt = 0, setWhich = 0;
		fprintf(of,"SeqID\tuniqKmers\tK_novelty\tKspace_sat\tAdj_dist\n");
		fprintf(of,"%s\t%u\t%.9f\t%f\t%f\n",HeadPack[whichMaxPop],maxPop,1.0,
			(double)maxPop/Kspace_denom, 1.);
		
		for (uint32_t z = 1; z < N; ++z) {
			uint32_t oldSetPop = setPop;
			#pragma omp parallel
			{
				//uint64_t *TestFP = calloc(FPSZ,sizeof(*TestFP));
				uint32_t threadPop = setPop, threadInt = setInt,
					threadWhich = 0;
				// Find the bug that raises the setPop the most
				#pragma omp for schedule(dynamic)
				for (uint32_t i = 0; i < N; ++i) {
					if (Mask[i]) continue; // skip those already in the pot
					//for (uint32_t i = 0; i < FPSZ; ++i) TestFP[i] = SetFP[i];
					uint32_t unn = 0, its = 0;
					uint64_t *Agg = ProfPack[i];
					#pragma omp simd reduction(+:its)
					for (uint32_t j = 0; j < FPSZ; ++j) 
						its += _mm_popcnt_u64(SetFP[j] & Agg[j]);
					unn = oldSetPop + Pops[i] - its;
					if (unn > threadPop) {
						threadPop = unn, threadWhich = i;
						threadInt = its;
					}
				}
				#pragma omp critical
				if (threadPop > setPop || (threadPop == setPop && threadInt < setInt))
					setPop = threadPop, setInt = threadInt, setWhich = threadWhich;
			}
			uint64_t *Agg = ProfPack[setWhich];
			#pragma omp parallel for
			for (uint32_t i = 0; i < FPSZ; ++i) SetFP[i] |= Agg[i]; // dump into master bin
			
			// expectation management
			uint32_t h = oldSetPop, l = Pops[setWhich], t;
			if (l > h) t = l, l = h, h = t;
			uint32_t its = setInt, unn = setPop;
			double chance = rspace * (double)l * (double)h;
			chance = chance > its ? its : chance;
			double denom = (double)unn * (double)l/h;
			its -= chance; denom -= chance;
			denom = denom > 1 ? denom : 1.; 
			double sim = (double)its / denom, dist = 1.0 - sim;
			dist = 1. - (1. + K_recip*log(2*sim/(1. + sim)));
			dist = dist > 1 ? 1 : dist;

			fprintf(of,"%s\t%u\t%.9f\t%f\t%f\n",HeadPack[setWhich],setPop,
				1.0-(double)oldSetPop/setPop,(double)setPop/Kspace_denom,dist);
			Set[numSet++] = setWhich; Mask[setWhich] = 1;
			
			if (!noCut && 1.0-(double)oldSetPop/setPop < 0.001) {
				printf("Reached .001 information gain at %u.\n",numSet);
				break;
			} else if (oldSetPop == setPop) {
				printf("Reached 0 information gain at %u.\n",numSet);
				break;
			}
		}
		
		printf("Finished set clustering [%f s].\n",omp_get_wtime() - wtime);
		exit(0);
	}
	
	if (doTree) {
		typedef union {uint32_t i; float f;} if_t;
		void *DD_init, *R_init, *X_init;
		uint64_t NP8 = N + 16, NSZ = N*NP8/2+15;
		float *DD = malloc_a(64,NSZ*sizeof(*DD),&DD_init), 
			**D = malloc(N*sizeof(*D)),
			*X = malloc_a(64,NP8*sizeof(*X),&X_init),
			*R = malloc_a(64,NP8*sizeof(*R),&R_init);
		if (!(DD_init && D && X_init && R_init)) 
			{fputs("OOM:DMat\n",stderr); exit(3);}
		*D = DD; for (uint32_t j = 1; j < N; ++j) 
			D[j] = D[j-1] + j-1 + (15 & (16 - ((j-1) & 15)));
		if (D[N-1] + N - 2 + (15 & (16 - ((N-2) & 15))) > DD + NSZ) 
			{puts("ERR 57"); exit(57);}
		#pragma omp parallel for
		for (uint64_t i = 0; i < NSZ; ++i)
			DD[i] = FLT_MAX; // Pad the end with infinity 
		memset(R+N-1,0,16*sizeof(*R)); 
		
		for (uint32_t j = 1; j < N; ++j) {
			uint64_t *F = ProfPack[j]; float *FC = D[j];
			#pragma omp parallel for schedule(dynamic)
			for (uint32_t k = 0; k < j; ++k) {
				uint64_t its = 0, *S = ProfPack[k];
				for (uint32_t z = 0; z < FPSZ; ++z) 
					its += _mm_popcnt_u64(F[z] & S[z]);
				FC[k] = calc_dist(Pops[j],Pops[k],GC[j],GC[k],its,Pops[j]+Pops[k]-its,
					doChance,doGC,ani,direct,adj,global,local);
			}
		}
		
		free(ProfDump); free(ProfPack); free(Pops);
		printf("Calculated distance matrix [%f]\n",omp_get_wtime()-wtime);
		
		// Per-OTU net divergence, M init
		wtime = omp_get_wtime();
		#pragma omp parallel for
		for (uint32_t i = 0; i < N; ++i) {
			float sum = 0, *Di = D[i];
			#pragma omp simd reduction(+:sum) aligned(Di)
			for (uint32_t j = 0; j < i; ++j) sum += Di[j];
			for (uint32_t j = i+1; j < N; ++j) sum += D[j][i];
			X[i] = sum;
		}
		uint32_t *HeadLens = malloc(N*sizeof(*HeadLens));
		#pragma omp parallel for
		for (uint32_t i = 0; i < N; ++i) HeadLens[i] = strlen(HeadPack[i]);
		
		for (uint32_t n = N; n > 2; --n) {
			float d = 1./(n-2);
			#pragma omp simd aligned(R,X)
			for (uint32_t i = 0; i < n; ++i) R[i] = X[i] * d;
			#ifndef __AVX__
			uint32_t TI[THREADS], TJ[THREADS];
			float min = INFINITY;
			uint32_t mi = 0, mj = 0; // to -1 if last
			#pragma omp parallel
			{
				uint32_t tmi, tmj, tid = omp_get_thread_num();
				float tmin = INFINITY;
				#pragma omp for reduction(min:min) schedule(guided)
				for (uint32_t i = 1; i < n; ++i) {
					float *Di = __builtin_assume_aligned(D[i],64), 
						Ri = R[i];
					for (uint32_t j = 0; j < i; ++j) {
						float t = Di[j] - (Ri + R[j]);
						if (t <= min) min = tmin = t, tmi = i, tmj = j;
					}
				}
				TI[tid] = tmin == min ? tmi : 0, TJ[tid] = tmj;
			}
			for (uint32_t i = 0; i < THREADS; ++i) //if (TI[i]) // rm, mi 0, < to >
				if (TI[i] > mi) mi = TI[i], mj = TJ[i];
			#else
			__m256 gmin = _mm256_set1_ps(FLT_MAX);
			__m256 grow = _mm256_castsi256_ps(_mm256_set1_epi32(0)), gcol = grow;
			__m256 TMIN[THREADS], TI[THREADS], TJ[THREADS];
			#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				__m256 tmin = _mm256_set1_ps(FLT_MAX);
				__m256 trow = _mm256_castsi256_ps(_mm256_set1_epi32(0)), tcol = trow;
				#pragma omp for schedule(dynamic,2)
				for (uint32_t i = 1; i < n; ++i) {
					__m256 rmin = _mm256_set1_ps(FLT_MAX);
					__m256 rcol = _mm256_castsi256_ps(_mm256_set1_epi32(0));
					
					float *Di = D[i]; 
					__m256 Ri = _mm256_set1_ps(R[i]);
					for (uint32_t j = 0; j < i; j+=16) {
						__m256 t = _mm256_sub_ps(_mm256_load_ps(Di+j),
							_mm256_add_ps(Ri,_mm256_load_ps(R+j)));
						__m256 m = _mm256_min_ps(t,rmin);
						__m256 c = _mm256_cmp_ps(m,t,_CMP_EQ_OQ);
						rmin = m;
						
						__m256i ix = _mm256_set1_epi32(j);
						rcol = _mm256_blendv_ps(rcol,_mm256_castsi256_ps(ix),c);
						
						uint32_t j8 = j + 8;
						t = _mm256_sub_ps(_mm256_load_ps(Di+j8),
							_mm256_add_ps(Ri,_mm256_load_ps(R+j8)));
						m = _mm256_min_ps(t,rmin);
						c = _mm256_cmp_ps(m,t,_CMP_EQ_OQ);
						rmin = m;
						ix = _mm256_set1_epi32(j8);
						rcol = _mm256_blendv_ps(rcol,_mm256_castsi256_ps(ix),c);
					}
					__m256 m = _mm256_min_ps(rmin,tmin);
					__m256 c = _mm256_cmp_ps(m,rmin,_CMP_EQ_OQ);
					tmin = m;
					tcol = _mm256_blendv_ps(tcol,rcol,c);
					trow = _mm256_blendv_ps(trow,_mm256_castsi256_ps(_mm256_set1_epi32(i)),c);
				}
				TMIN[tid] = tmin, TI[tid] = trow, TJ[tid] = tcol;
			}
			// Cross reduction
			for (int i = 0; i < THREADS; ++i) {
				__m256 tmin = TMIN[i], trow = TI[i], tcol = TJ[i];
				__m256 lt = _mm256_cmp_ps(tmin,gmin,_CMP_LT_OQ),
					eq = _mm256_cmp_ps(tmin,gmin,_CMP_EQ_OQ),
					r_gt = _mm256_cmp_ps(_mm256_cvtepi32_ps(_mm256_castps_si256(trow)),
						_mm256_cvtepi32_ps(_mm256_castps_si256(grow)),_CMP_GT_OQ),
					c = _mm256_or_ps(lt,_mm256_and_ps(eq,r_gt));
				gmin = _mm256_min_ps(tmin,gmin); 
				gcol = _mm256_blendv_ps(gcol,tcol,c);
				grow = _mm256_blendv_ps(grow,trow,c);
			}
			// Self reduction
			__m128i c0 = _mm256_extractf128_si256(_mm256_castps_si256(gcol),0),
				c1 = _mm256_extractf128_si256(_mm256_castps_si256(gcol),1),
				r0 = _mm256_extractf128_si256(_mm256_castps_si256(grow),0),
				r1 = _mm256_extractf128_si256(_mm256_castps_si256(grow),1);
			__m128 m0 = _mm256_extractf128_ps(gmin,0),
				m1 = _mm256_extractf128_ps(gmin,1);
			c0 = _mm_add_epi32(c0,_mm_setr_epi32(0,1,2,3));
			c1 = _mm_add_epi32(c1,_mm_setr_epi32(4,5,6,7));
			__m128 ms = _mm_min_ps(m0,m1);
			ms = _mm_min_ps(ms,_mm_castsi128_ps(_mm_srli_si128(
				_mm_castps_si128(ms),8)));
			ms = _mm_min_ps(ms,_mm_castsi128_ps(_mm_srli_si128(
				_mm_castps_si128(ms),4)));
			float min = ((if_t){.i=_mm_extract_epi32(_mm_castps_si128(ms),0)}).f;
			ms = _mm_set1_ps(min); //broadcast
			__m128i x0 = _mm_castps_si128(_mm_cmpeq_ps(m0,ms)), 
				x1 = _mm_castps_si128(_mm_cmpeq_ps(m1,ms));
			r0 = _mm_and_si128(x0,r0); r1 = _mm_and_si128(x1,r1);
			__m128i mr = _mm_max_epu32(r0,r1);
			mr = _mm_max_epu32(mr,_mm_srli_si128(mr,8));
			mr = _mm_max_epu32(mr,_mm_srli_si128(mr,4));
			uint32_t mi = _mm_extract_epi32(mr,0);
			mr = _mm_set1_epi32(mi);
			x0 = _mm_cmpeq_epi32(r0,mr), x1 = _mm_cmpeq_epi32(r1,mr);
			c0 = _mm_and_si128(x0,c0); c1 = _mm_and_si128(x1,c1);
			__m128i mc = _mm_max_epu32(c0,c1);
			mc = _mm_max_epu32(mc,_mm_srli_si128(mc,8));
			mc = _mm_max_epu32(mc,_mm_srli_si128(mc,4));
			uint32_t mj = _mm_extract_epi32(mc,0);
			#endif
			
			if (mj >= mi || mj >= n) {printf("ERR MJ: mi %u, mj %u\n",mi, mj);}
			if (!mi || mi >=n) printf("ERR MI: mi %u, mj %u\n", mi, mj);
			float md = D[mi][mj], b1 = 0.5f * (md + (R[mi]-R[mj])), 
				b2 = md - b1, md2 = 0.5f * md; // new branch lengths
			//if () // allow option to not make rooted
			
			// Both node lengths, two colons, a comma, two bounding parentheses, 
			// 2 7-char numbers (controlled with %.5f), 2 neg signs, and a null = 22
			uint32_t nlen = HeadLens[mi]+HeadLens[mj] + 32;
			char *Us = malloc(nlen); 
			if (!Us) {fputs("OOM:Us\n",stderr); exit(3);}
			HeadLens[mj] = sprintf(Us,"(%s:%.5f,%s:%.5f)",HeadPack[mi],b1,HeadPack[mj],b2);
			free(HeadPack[mi]), free(HeadPack[mj]);
			HeadPack[mj] = Us; 
			
			// Redefine R to be the common term (D[mi][i] + D[mj][i])/2
			float *Dmi = D[mi], *Dmj = D[mj], Xu = 0.f;
			#pragma omp parallel
			{
				#pragma omp for simd aligned(Dmi,Dmj,R) nowait
				for (uint32_t i = 0; i < mj; ++i) 
					R[i] = 0.5f * (Dmi[i] + Dmj[i]);
				#pragma omp for nowait
				for (uint32_t i = mj+1; i < mi; ++i) 
					R[i] = 0.5f * (Dmi[i] + D[i][mj]);
				#pragma omp for nowait
				for (uint32_t i = mi+1; i < n; ++i) 
					R[i] = 0.5f * (D[i][mi] + D[i][mj]);
			}
			R[mi] = R[mj] = md2;
			#pragma omp simd aligned(X,R) reduction(+:Xu)
			for (uint32_t i = 0; i < n; ++i) 
				X[i] -= R[i] + md2,
				Xu += R[i] = R[i] - md2; 
			X[mj] = Xu;
			
			#pragma omp parallel
			{
				#pragma omp for simd aligned(Dmi,R) nowait
				for (uint32_t i = 0; i < mj; ++i) Dmj[i] = R[i];
				#pragma omp for nowait
				for (uint32_t i = mj+1; i < n; ++i) D[i][mj] = R[i];
			}
			
			if (mi < n - 1) { // Compact D: move last row into mi's slot
				float *LR = D[n-1];
				#pragma omp parallel
				{
					#pragma omp for simd aligned(Dmi,LR) nowait
					for (uint32_t i = 0; i < mi; ++i) Dmi[i] = LR[i]; // vec
					#pragma omp for nowait
					for (uint32_t i = mi+1; i < n; ++i) D[i][mi] = LR[i];
				}
				X[mi] = X[n-1];
				HeadPack[mi] = HeadPack[n-1];
				HeadLens[mi] = HeadLens[n-1];
			}
			//printf("Finished iteration %u [%f]\n",n,omp_get_wtime()-wtime);
			//wtime = omp_get_wtime();
		}
		fprintf(of, "(%s:%.5f,%s:%.5f);\n", HeadPack[0], D[1][0]/2.f, HeadPack[1], D[1][0]/2.f);
		printf("Finished tree construction [%f]\n",omp_get_wtime()-wtime);
		exit(0); 
	}
	
	// Write the distance matrix
	wtime = omp_get_wtime();
	if (!doMin2) {
		for (uint32_t j = 0; j < N; ++j) fprintf(of,"\t%s",HeadPack[j]);
		fputc('\n',of);
	}
	float *FC = malloc(N*sizeof(*FC));
	// Do F2 tally
	if (f2) {
		uint64_t nF2 = 0;
		uint64_t bufSz = INT32_MAX-1;
		uint8_t *line = malloc(bufSz + 1);
		double *DistQ = malloc(N*sizeof(*DistQ));
		
		uint32_t K1_2 = K1<<1, shifty = (32 - 2*K);
		
		uint8_t **QBones = malloc(omp_get_max_threads()*sizeof(*QBones));
		#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			QBones[tid] = calloc((uint64_t)1 << (2*K-3),1);
		}
		
		while (fgets(line,bufSz,f2)) {
			int hlen = strlen(line);
			line[hlen-1] = 0;
			fprintf(of,"%s",line+1);

			fgets(line,bufSz,f2);
			int len = strlen(line);
			line[len-1] = 0;

			#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				uint8_t *QBone = QBones[tid];

				#pragma omp for schedule(dynamic,1)
				for (uint32_t i = 0; i < N; ++i) {
					uint8_t *R = (uint8_t *)ProfPack[i];
					int its = 0; int nGC = 0, nAmbig = 0;
					uint32_t w = 0, nGood = 0, wrc = 0, tot = 0;
					for (int j = 0; j < len; ++j) {
						w >>= 2;
						uint8_t x = CONV[line[j] & 31];
						if (x > 3) x = 0, nGood=0, ++nAmbig;
						else ++nGood, nGC += (x==1 || x==2);
						w |= x << K1_2;
						uint32_t cand = w << shifty >> shifty, rcand;
						if (doRC) wrc <<= 2, wrc |= 3-x,
							rcand = wrc << shifty >> shifty,
							cand = cand < rcand ? cand : rcand;
						if (nGood >= K && !(QBone[cand >> 3] & (1 << (cand & 7))) ) {
							QBone[cand >> 3] |= 1 << (cand & 7);
							++tot;
							its += (R[cand >> 3] & (1 << (cand & 7))) != 0;
						}
					}
					w = 0, wrc = 0;
					for (int j = 0; j < len; ++j) {
						w >>= 2;
						uint8_t x = CONV[line[j] & 31] & 3;
						w |= x << K1_2;
						uint32_t cand = w << shifty >> shifty, rcand;
						if (doRC) wrc <<= 2, wrc |= 3-x,
							rcand = wrc << shifty >> shifty,
							cand = cand < rcand ? cand : rcand;
						QBone[cand >> 3] = 0;
					}
					int comboPop = Pops[i] + tot;
					if (its > comboPop) its = comboPop;
					int unn = comboPop - its;
					double gc = doGC ? (double)nGC / (len-nAmbig) : 0.5;
					//printf("GC = %f\n",gc);
					DistQ[i]=calc_dist(Pops[i],tot,GC[i],gc,its,unn,doChance,doGC,ani,direct,adj,global,local);
				}
			}
			if (doMin2) {
				double min = INFINITY, min2 = INFINITY; 
				int minIx = 0, minIx2 = 0;
				for (int i = 0; i < N; ++i) {
					if (DistQ[i] < min) 
						min2 = min, minIx2 = minIx,
						min = DistQ[i], minIx = i;
					else if (DistQ[i] < min2)
						min2 = DistQ[i], minIx2 = i;
				}
				fprintf(of,"\t%s\t%f\t%s\t%f",HeadPack[minIx],min,HeadPack[minIx2],min2);
			} else for (int i = 0; i < N; ++i) fprintf(of,"\t%.5g",DistQ[i]);
			fputc('\n',of);
			++nF2;
		}
		printf("Processed %lu query sequences [%f]\n",nF2,omp_get_wtime()-wtime);
		exit(0);
	}

	for (uint32_t j = 0; j < N; ++j) {
		uint64_t *F = ProfPack[j];
		fprintf(of,"%s",HeadPack[j]);
		#pragma omp parallel for schedule(dynamic)
		for (uint32_t k = 0; k < j; ++k) {
			uint64_t *S = ProfPack[k];
			uint32_t its = 0;
			for (uint32_t z = 0; z < FPSZ; ++z)
				its += _mm_popcnt_u64(F[z] & S[z]);
			FC[k] = calc_dist(Pops[k],Pops[j],GC[k],GC[j],its,Pops[k]+Pops[j]-its,
				doChance,doGC,ani,direct,adj,global,local);
		}
		for (uint32_t k = 0; k < j; ++k) fprintf(of,"\t%.5g",FC[k]);
		fputs("\n",of);
	}
	printf("Finished distance matrix write [%f]\n",omp_get_wtime()-wtime);
}
