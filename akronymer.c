#define VER "v0.95"
#define USAGE "usage: aKronyMer inseqs.lin.fna output [K] [HEUR[0-9]] [ADJ] [GLOBAL/DIRECT] [TREE]"
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <float.h>
#include <math.h>
const uint8_t CONV[32] = {7,0,7,1,7,7,7,2,7,7,7,7,7,7,7,7,7,7,7,7,3,3,7,7,7,7,7,7,7,7,7,7};
__m128i MSK[16];

void * malloc_a(size_t algn, size_t size, void **oldPtr) {
	uintptr_t mask = ~(uintptr_t)(algn - 1);
	*oldPtr = malloc(size+algn-1);
	return (void *)(((uintptr_t)*oldPtr+algn-1) & mask);
}

void main(int argc, char *argv[]) {
	puts("This is aKronyMer " VER " by Gabe.");
	MSK[15] = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
	MSK[14] = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0);
	MSK[13] = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0);
	MSK[12] = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0);
	MSK[11] = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0);
	MSK[10] = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0);
	MSK[9] =  _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0);
	MSK[8] =  _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0);
	MSK[7] =  _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0);
	MSK[6] =  _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0);
	MSK[5] =  _mm_setr_epi8(-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0);
	MSK[4] =  _mm_setr_epi8(-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0);
	MSK[3] =  _mm_setr_epi8(-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0);
	if (argc < 3) {puts(USAGE); exit(1);}
	FILE *fp = fopen(argv[1],"rb");
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
	int doTree = argc >= 4 && !strcmp(argv[argc-1],"TREE"); 
	argc -= doTree;
	int global = argc >= 4 && !strcmp(argv[argc-1],"GLOBAL");
	argc -= global;
	int direct = argc >= 4 && !strcmp(argv[argc-1],"DIRECT");
	argc -= direct;
	int adj = argc >= 4 && !strcmp(argv[argc-1],"ADJ");
	argc -= adj;
	char *hr = 0, hlv = 0;
	int heur = argc >= 4 && (hr=strstr(argv[argc-1],"HEUR"));
	argc -= heur;
	if (hr) hlv = atoi(hr+4);
	printf("Goal: output %s %s %s\n", adj ? "adjusted" : "raw", global ? "global"
		: direct ? "direct" : "glocal", doTree ? "tree" : "distance matrix");
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
	printf("Avg. length: %lu, max = %lu. Sugg. K = %u\n", totL/N, maxL, sugK);
	if (N < 2) {fputs("Sorry, need > 1 sequence!\n",stderr); exit(1);}
	uint32_t K = argc > 3 ? atoi(argv[3]) : sugK, PSZ, K1, FPSZ, FPSZH;
	K = K < 16 ? K : 16, K = K > 4 ? K : 4, K1 = K - 1;
	PSZ = 2*K-3; FPSZ = 1 << (PSZ-3), FPSZH = FPSZ >> hlv; 
	FPSZH += !FPSZH;
	printf("Running with K = %u [H = %u]\n",K,FPSZH);
	HeadPack = realloc(HeadPack,N*sizeof(*HeadPack));
	Lens = realloc(Lens,N*sizeof(*Lens));
	uint32_t *Pops = malloc(N*sizeof(*Pops));
	uint64_t **ProfPack = malloc(N*sizeof(*ProfPack));
	uint64_t *ProfDump = calloc((uint64_t)N*FPSZ,sizeof(*ProfDump));
	if (!ProfPack || !Pops || !ProfDump) {fputs("OOM:Dump\n",stderr); exit(3);}
	for (uint64_t j = 0; j < N; ++j) ProfPack[j] = ProfDump + j*FPSZ;
	i = fx = 0, head = headO, seq = seqO;
	rewind(fp);
	while (head = fgets(head,HEADLEN,fp)) {
		seq = fgets(seq,SEQLEN,fp);
		seq[Lens[i]-1] = 0;
		uint8_t *P = (uint8_t *)ProfPack[fx];
		uint32_t fp = 0, len = Lens[i];
		uint16_t re = (uint16_t)-1 << (16 - K);
		#pragma omp parallel
		{
			#pragma omp for schedule(dynamic,192)
			for (uint32_t j = K1; j < len; ++j) {
				#ifndef __SSE4_2__
				#define _mm_popcnt_u64 __builtin_popcountll
				char *s = seq + j - K1;
				uint32_t w = 0;
				for (uint32_t k = 0; k < K; ++k) {
					uint32_t x = CONV[s[k] & 31];
					if (x > 3u) goto ENDR;
					w |= x << (k << 1);
				}
				#else
				__m128i s = _mm_lddqu_si128((void*)(seq+j-K1));
				s = _mm_and_si128(s,_mm_set1_epi8(31));
				__m128i xa = _mm_cmpeq_epi8(s,_mm_set1_epi8(1));
				__m128i xc = _mm_cmpeq_epi8(s,_mm_set1_epi8(3));
				__m128i xg = _mm_cmpeq_epi8(s,_mm_set1_epi8(7));
				__m128i xt = _mm_cmpeq_epi8(s,_mm_set1_epi8(20));
				__m128i r1 = _mm_or_si128(xa,xc), r2 = _mm_or_si128(xg,xt);
				r1 = _mm_or_si128(r1,r2);
				uint16_t r = (uint16_t)_mm_movemask_epi8(r1) << (16-K);
				if (r != re) continue;
				__m128i zc = _mm_and_si128(xc,_mm_set1_epi8(1));
				__m128i zg = _mm_and_si128(xg,_mm_set1_epi8(2));
				__m128i zt = _mm_and_si128(xt,_mm_set1_epi8(3));
				zc = _mm_or_si128(zc,zg);
				__m128i a = _mm_or_si128(zc,zt);
				a = _mm_and_si128(a,MSK[K1]);
				
				__m128i s1 = _mm_cvtepu8_epi32(a),
					s2 = _mm_cvtepu8_epi32(_mm_srli_si128(a,4)),
					s3 = _mm_cvtepu8_epi32(_mm_srli_si128(a,8)),
					s4 = _mm_cvtepu8_epi32(_mm_srli_si128(a,12));
				s1 = _mm_mullo_epi32(s1,_mm_setr_epi32(1,1<<2,1<<4,1<<6));
				s2 = _mm_mullo_epi32(s2,_mm_setr_epi32(1<<8,1<<10,1<<12,1<<14));
				s3 = _mm_mullo_epi32(s3,_mm_setr_epi32(1<<16,1<<18,1<<20,1<<22));
				s4 = _mm_mullo_epi32(s4,_mm_setr_epi32(1<<24,1<<26,1<<28,1<<30));
				s1 = _mm_or_si128(s1,s2);
				s2 = _mm_or_si128(s3,s4);
				s1 = _mm_or_si128(s1,s2);
				s1 = _mm_or_si128(s1,_mm_srli_si128(s1,8));
				s1 = _mm_or_si128(s1,_mm_srli_si128(s1,4));
				uint32_t w = _mm_extract_epi32(s1,0); 
				#endif
				
				#pragma omp atomic
				P[w >> 3] |= 1 << (w & 7); 
				ENDR:NULL;
			}
			uint64_t *Agg = ProfPack[fx];
			#pragma omp for reduction(+:fp)
			for (uint32_t j = 0; j < FPSZ; ++j)
				fp += _mm_popcnt_u64(Agg[j]);
		}
		printf("[%u (%u)] L = %u, Density = %u [%f], Entropy = %f\n",i, fx, len, fp, 
			(double)fp/((uint64_t)1 << (2*K)),(double)fp/len);
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
	printf("Done parsing %u (%u cls) sequences [%f]\n",i,N,omp_get_wtime()-wtime);
	free(seqO); free(headO); free(Lens); fclose(fp);
	if (N < i) ProfDump = realloc(ProfDump,(uint64_t)N*(1 << PSZ)),
		ProfPack = realloc(ProfPack,N*sizeof(*ProfPack));

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
		wtime = omp_get_wtime();
		if (K > 4) {
			#pragma omp parallel for schedule(guided)
			for (uint32_t j = 1; j < N; ++j) {
				uint64_t *F = ProfPack[j];
				for (uint32_t k = 0; k < j; ++k) {
					uint64_t its = 0, *S = ProfPack[k];
					for (uint32_t z = 0; z < FPSZ; z+=8) {
						uint64_t a1 = F[z] & S[z], a2 = F[z+1] & S[z+1], 
							a3 = F[z+2] & S[z+2], a4 = F[z+3] & S[z+3], 
							a5 = F[z+4] & S[z+4], a6 = F[z+5] & S[z+5], 
							a7 = F[z+6] & S[z+6], a8 = F[z+7] & S[z+7],
							p1 = _mm_popcnt_u64(a1), p2 = _mm_popcnt_u64(a2),
							p3 = _mm_popcnt_u64(a3), p4 = _mm_popcnt_u64(a4),
							p5 = _mm_popcnt_u64(a5), p6 = _mm_popcnt_u64(a6),
							p7 = _mm_popcnt_u64(a7), p8 = _mm_popcnt_u64(a8);
						its += p1+p2+p3+p4+p5+p6+p7+p8; 
					}
					float denom;
					if (direct) denom = Pops[j] + Pops[k] - its;
					else { 
						uint32_t h, l;
						if (Pops[j] > Pops[k]) h = Pops[j], l = Pops[k];
						else h = Pops[k], l = Pops[j];
						denom = global ? h : l;
					}
					D[j][k] = 1.f - (float)its / denom;
				}
			}
		} else { // K = 4 special case (no z loop)
			#pragma omp parallel for schedule(guided)
			for (uint32_t j = 1; j < N; ++j) {
				uint64_t *F = ProfPack[j];
				for (uint32_t k = 0; k < j; ++k) {
					uint64_t *S = ProfPack[k];
					uint64_t a1 = F[0] & S[0], a2 = F[1] & S[1], 
						a3 = F[2] & S[2], a4 = F[3] & S[3], 
						p1 = _mm_popcnt_u64(a1), p2 = _mm_popcnt_u64(a2),
						p3 = _mm_popcnt_u64(a3), p4 = _mm_popcnt_u64(a4);
					uint64_t its = p1+p2+p3+p4; 
					float denom;
					if (direct) denom = Pops[j] + Pops[k] - its;
					else { 
						uint32_t h, l;
						if (Pops[j] > Pops[k]) h = Pops[j], l = Pops[k];
						else h = Pops[k], l = Pops[j];
						denom = global ? h : l;
					}
					D[j][k] = 1.f - (float)its / denom;
				}
			}
		}
		if (adj) { // LBA fix 
			float s = (uint64_t)1 << (K << 1), s_r = 1.f/s;
			#pragma omp parallel for schedule(guided)
			for (uint32_t j = 1; j < N; ++j) 
				for (uint32_t k = 0; k < j; ++k) {
					float nu;
					if (direct) nu = (float)Pops[j]*Pops[k]/(Pops[j]+Pops[k]);
					else {
						uint32_t h, l;
						if (Pops[j] > Pops[k]) h = Pops[j], l = Pops[k];
						else h = Pops[k], l = Pops[j];
						nu = global ? l : h;
					}
					float rd = 1.f - nu*s_r;
					D[j][k] = D[j][k] >= rd ? 1 : D[j][k]/rd;
					D[j][k] = D[j][k] <= .9999546f ? -logf(1-D[j][k]) : 10.f;
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
						Ri = __builtin_assume_aligned(R[i],64);
					for (uint32_t j = 0; j < i; ++j) {
						float t = Di[j] - (Ri + R[j]);
						if (t <= min) min = tmin = t, tmi = i, tmj = j;
					}
				}
				TI[tid] = tmin == min ? tmi : 0, TJ[tid] = tmj;
			}
			for (uint32_t i = 0; i < THREADS; ++i) //if (TI[i]) // rm, mi 0, < to >
				if (TI[i] > mi) mi = TI[i], mj = TJ[i];
			// #elif __AVX512F__
			// __m512 gmin = _mm512_set1_ps(FLT_MAX);
			// __m512i grow = _mm512_set1_epi32(0), gcol = grow;
			// __m512 TMIN[THREADS];
			// __m512i TI[THREADS], TJ[THREADS];
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
		exit(1); 
	}
	
	for (uint32_t j = 0; j < N; ++j) fprintf(of,"\t%s",HeadPack[j]);
	fputc('\n',of);
	float *FC = malloc(N*sizeof(*FC)),
		s = (uint64_t)1 << (K << 1), s_r = 1.f/s;
	for (uint32_t j = 0; j < N; ++j) {
		uint64_t *F = ProfPack[j];
		fprintf(of,"%s",HeadPack[j]);
		#pragma omp parallel for
		for (uint32_t k = 0; k < j; ++k) {
			uint64_t *S = ProfPack[k];
			uint32_t its = 0;
			for (uint32_t z = 0; z < FPSZ; ++z)
				its += _mm_popcnt_u64(F[z] & S[z]);
			float denom;
			if (direct) denom = Pops[j] + Pops[k] - its;
			else { 
				uint32_t h, l;
				if (Pops[j] > Pops[k]) h = Pops[j], l = Pops[k];
				else h = Pops[k], l = Pops[j];
				denom = global ? h : l;
			}
			FC[k] = 1.f - (float)its / denom;
			if (adj) { // LBA fix 
				float nu;
				if (direct) nu = (float)Pops[j]*Pops[k]/(Pops[j]+Pops[k]);
				else {
					uint32_t h, l;
					if (Pops[j] > Pops[k]) h = Pops[j], l = Pops[k];
					else h = Pops[k], l = Pops[j];
					nu = global ? l : h;
				}
				float rd = 1.f - nu*s_r;
				FC[k] = FC[k] >= rd ? 1 : FC[k]/rd;
				FC[k] = FC[k] <= .9999546f ? -logf(1-FC[k]) : 10.f;
			}
		}
		for (uint32_t k = 0; k < j; ++k) fprintf(of,"\t%.4f",FC[k]);
		fputs("\n",of);
		//fputs("\t1.000\n",of); // omit?
	}
}
