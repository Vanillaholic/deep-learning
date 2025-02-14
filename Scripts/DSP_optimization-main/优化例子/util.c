#include "util.h"
#include "config.h"



void SSC_COPY32(int *dst, int *src, int n)
{
	int i;

#ifndef SSC_COPY32_QDSP_OPT

	for(i=n;i--;)
		dst[i] = src[i];

#else    
	//long long tmp1;
	long long *pt1,*pt2;
	pt1=(long long*)src;
	pt2=(long long*)dst;
	for(i=0;i<n;i+=2)
	{
		*pt2++=*pt1++;

	}
#endif
}

void SSC_COPY16(short *dst, short *src, int n)
{

	int i;
	for(i=n;i--;)
		dst[i] = src[i];

}



void SSC_COPY8(unsigned char *dst, unsigned char *src, int n)
{

	int i;
	for(i=n;i--;)
		dst[i] = src[i];

}

void SSC_CLEAR(char *dst, int n)
{

	int i;
	for(i=n;i--;)
		dst[i] = 0;

}

