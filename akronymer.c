#define VER "v0.89"
#define USAGE "usage: aKronyMer inseqs.lin.fna outmatrix.tsv [K]"
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
const uint8_t CONV[32] = {7,0,7,1,7,7,7,2,7,7,7,7,7,7,7,7,7,7,7,7,3,3,7,7,7,7,7,7,7,7,7,7};
__m128i MSK[16];

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
	uint32_t K = argc > 3 ? atoi(argv[3]) : 16, PSZ, K1, FPSZ;
	K = K < 16 ? K : 16, K = K > 4 ? K : 4, K1 = K - 1;
	PSZ = 2*K-3; FPSZ = 1 << (PSZ-3);
	printf("Using K = %u\n",K);
	const uint32_t SEQLEN = 1 << 24, HEADLEN = 1 << 16;
	char *head = malloc(HEADLEN + 1), *headO = head,
		*seq = calloc(SEQLEN + 16,1), *seqO = seq;
	head[HEADLEN] = 0, seq[SEQLEN] = 0;
	size_t profSz = 10, profIx = 0, i = 0;
	char **HeadPack = malloc(profSz*sizeof(*HeadPack));
	void **ProfPack = malloc(profSz*sizeof(*ProfPack));
	uint32_t *Pops = malloc(profSz*sizeof(*Pops));
	if (!(head && seq && HeadPack && ProfPack))
		{fputs("OOM:Init\n",stderr); exit(3);}
	while (head = fgets(head,HEADLEN,fp)) {
		char *h = strchr(head,'\n');
		if (!h || *head != '>') {fprintf(stderr,"ERROR: head %u\n",i); exit(2);}
		HeadPack[i] = malloc(h - head);
		if (!HeadPack[i]) {fputs("OOM:HeadPack_i\n",stderr); exit(3);}
		*h = 0, memcpy(HeadPack[i],head+1,h-head);
		seq = fgets(seq,SEQLEN,fp);
		if (!seq) {fprintf(stderr,"ERROR: sequence ln %u\n",i); exit(2);}
		uint32_t len = strlen(seq); // TODO: instead of just strlen, also convert as you go?
		len -= seq[len-1] == '\n';
		seq[len-1] = 0;
		void *Prof = calloc(1 << PSZ,1); // malloc(len);
		if (!Prof) {fputs("OOM:ProfPack_i\n",stderr); exit(3);}
		uint8_t *P = Prof; uint64_t *Agg = Prof;
		uint32_t fp = 0;
		uint16_t re = (uint16_t)-1 << (16 - K);
		#pragma omp parallel
		{
			#pragma omp for schedule(dynamic,192)
			for (uint32_t j = K1; j < len; ++j) {
				#ifndef __SSE3__
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
			#pragma omp for reduction(+:fp)
			for (uint32_t j = 0; j < FPSZ; ++j)
				fp += _mm_popcnt_u64(Agg[j]);
		}
		printf("[%u] L = %u, Density = %u [%f], Entropy = %f\n",i, len, fp, 
			(double)fp/((uint64_t)1 << (2*K)),(double)fp/len);
		ProfPack[i] = Prof;
		Pops[i] = fp;
		if (++i == profSz) {
			HeadPack = realloc(HeadPack,(profSz*=2)*sizeof(*HeadPack));
			ProfPack = realloc(ProfPack,profSz*sizeof(*ProfPack));
			Pops = realloc(Pops,profSz*sizeof(*Pops));
			if (!HeadPack || !ProfPack || !Pops) 
				{fputs("OOM:ProfPack\n",stderr); exit(3);}
		}
	}
	printf("Done parsing [%u sequences processed]\n",i);
	free(seqO); free(headO);
	/* float *Mat = malloc(i*i*sizeof(*Mat));
	for (uint32_t j = 0; j < i; ++j) {
		uint64_t *F = ProfPack[j];
		for (uint32_t k = j + 1; k < i; ++k) {
			uint64_t *S = ProfPack[k];
			uint32_t its = 0;
			for (uint32_t z = 0; z < FPSZ; ++z)
				its += _mm_popcnt_u64(F[z] & S[z]);
			
		}
	} */
	
	for (uint32_t j = 0; j < i; ++j) fprintf(of,"\t%s",HeadPack[j]);
	fputc('\n',of);
	float *FC = malloc(i*sizeof(*FC));
	for (uint32_t j = 0; j < i; ++j) {
		uint64_t *F = ProfPack[j];
		fprintf(of,"%s",HeadPack[j]);
		//for (uint32_t k = 0; k < j; ++k) fputc('\t',of);
		#pragma omp parallel for
		for (uint32_t k = 0; k < j; ++k) {
			uint64_t *S = ProfPack[k];
			uint32_t its = 0;
			for (uint32_t z = 0; z < FPSZ; ++z)
				its += _mm_popcnt_u64(F[z] & S[z]);
			FC[k] = (double)its/(Pops[j] < Pops[k] ? Pops[j] : Pops[k]);
		}
		
		for (uint32_t k = 0; k < j; ++k) fprintf(of,"\t%.4f",FC[k]);
		fputs("\t1.000\n",of);
		//fputc('\n',of);
	}
}
