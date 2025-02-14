#include "pitch_util.h"
#include "config.h"

#ifdef QDSP_INTRINSIC
#include <q6protos.h>
#endif

#ifndef ssc_pitch_xcorr_c2_QDSP_OPTI
int ssc_pitch_xcorr_c2(const short *_x, const short *_y, int *xcorr, int len, int max_pitch)
{
	int i;
	int maxcorr=1;
	
	int sum[4];

	for (i=0;i<max_pitch-3;i+=4)
	{
		sum[0] = sum[1] = sum[2] = sum[3] = 0;
		xcorr_kernel(_x, _y+i, sum, len);
		xcorr[i]=sum[0];
		xcorr[i+1]=sum[1];
		xcorr[i+2]=sum[2];
		xcorr[i+3]=sum[3];
#ifndef QDSP_INTRINSIC
		sum[0] = MAX32(sum[0], sum[1]);
		sum[2] = MAX32(sum[2], sum[3]);
		sum[0] = MAX32(sum[0], sum[2]);
		maxcorr = MAX32(maxcorr, sum[0]);
#else
		sum[0] = Q6_R_max_RR(sum[0], sum[1]);
		sum[2] = Q6_R_max_RR(sum[2], sum[3]);
		sum[0] = Q6_R_max_RR(sum[0], sum[2]);
		maxcorr = Q6_R_max_RR(maxcorr, sum[0]);
#endif
	}

	return maxcorr;
}
#else
/*
int ssc_pitch_xcorr_c2(const short *_x, const short *_y, int *xcorr, int len, int max_pitch)
{
	int i, j, maxcorr, sum;
	maxcorr = 1;
	for (i=0; i<max_pitch; i++)
	{
		sum = 0;
		for(j=0; j<len; j++)
		{
			sum += (int)_x[j]*_y[i+j];
		}
		xcorr[i] = sum;
		maxcorr = MAX32(maxcorr, sum);
	}
	return maxcorr;
}
*/
/*
int ssc_pitch_xcorr_c2(const short *_x, const short *_y, int *xcorr, int len, int max_pitch)
{
	int i, j;
	long long * vInput = (long long*)_y;
	long long * vCoeff = (long long*)_x;
	long long * vOutput = (long long*)xcorr;
	long long sum0, sum1, sum2, sum3;
	int max, maxcorr;
	
	maxcorr = 1;
	for (i = 0; i < (max_pitch>>2); i++)
	{
		sum0 = sum1 = sum2 = sum3 = 0;
		for (j = 0; j < (len>>2); j++)
		{
			long long vIn1 = vInput[i+j];
			long long vIn2 = vInput[i+j+1];
			long long curCoeff = vCoeff[j];
			long long curIn;
			curIn = vIn1;
			sum0 = Q6_P_vrmpyhacc_PP(sum0, curIn, curCoeff);
			curIn = Q6_P_valignb_PPI(vIn2, vIn1, 2);
			sum1 = Q6_P_vrmpyhacc_PP(sum1, curIn, curCoeff);
			curIn = Q6_P_valignb_PPI(vIn2, vIn1, 4);
			sum2 = Q6_P_vrmpyhacc_PP(sum2, curIn, curCoeff);
			curIn = Q6_P_valignb_PPI(vIn2, vIn1, 6);
			sum3 = Q6_P_vrmpyhacc_PP(sum3, curIn, curCoeff);
		}
		sum0 = Q6_P_combine_RR(sum1, sum0);
		sum2 = Q6_P_combine_RR(sum3, sum2);
		sum1 = Q6_P_vmaxw_PP(sum0, sum2);
		max = Q6_R_max_RR((int)sum1, (int)(sum1>>32));
		maxcorr = Q6_R_max_RR(maxcorr, max);
		vOutput[2*i] = sum0;
		vOutput[2*i+1] = sum2;
	}
	
	return maxcorr;
}
*/
#endif


int ssc_pitch_xcorr_c1(short *_x, short *_y, int *xcorr)
{



#ifndef ssc_pitch_xcorr_c1_QDSP_OPTI

	short j = 940;
	int xcorr_0 = 0;
	int xcorr_1 = 0;
	int xcorr_2 = 0;
	int xcorr_3 = 0;
	int xcorr_4 = 0;



	do
	{
		xcorr_0 = MAC16_16(xcorr_0, *_x, *_y++);
		xcorr_1 = MAC16_16(xcorr_1, *_x, *_y);
		xcorr_2 = MAC16_16(xcorr_2, *_x, *(_y + 1));
		xcorr_3 = MAC16_16(xcorr_3, *_x, *(_y + 2));
		xcorr_4 = MAC16_16(xcorr_4, *_x++, *(_y + 3));
	} while (--j);

	xcorr[0] = xcorr_0;
	xcorr[1] = xcorr_1;
	xcorr[2] = xcorr_2;
	xcorr[3] = xcorr_3;
	xcorr[4] = xcorr_4;

#else
   long long *ptx;
   long long x0, x1;
   int i;
   	long long xcorr_0 = 0;
	long long xcorr_1 = 0;
	long long xcorr_2 = 0;
	long long xcorr_3 = 0;
	long long xcorr_4 = 0;

   ptx = (long long *)_x;
   
   x0 = *ptx++;

   for(i=0; i<940; i+=4)
   {
	    long long tmpll;
	    x1 = *ptx++;
	    xcorr_0 = Q6_P_vrmpyhacc_PP(xcorr_0, x0, x0);

		tmpll = ((x0 >> 16)&0xffffffffffff)|((x1&0xffff)<<48);
	//	tmpll = Q6_P_valignb_PPI(x1,x0,2);
		xcorr_1 = Q6_P_vrmpyhacc_PP(xcorr_1, x0, tmpll);

	    tmpll = ((x0 >> 32)&0xffffffff)|((x1&0xffffffff)<<32);
	//	tmpll = Q6_P_valignb_PPI(x1,x0,4);
		xcorr_2 = Q6_P_vrmpyhacc_PP(xcorr_2, x0, tmpll);

//		tmpll = Q6_P_valignb_PPI(x1,x0,6);
		tmpll = ((x0 >> 48)&0xffff)|((x1&0xffffffffffff)<<16);
		xcorr_3 = Q6_P_vrmpyhacc_PP(xcorr_3, x0, tmpll);

		xcorr_4 = Q6_P_vrmpyhacc_PP(xcorr_4, x0, x1);

		x0 = x1;

   }
   /*
    fcheck = fopen("E:\\2019_QDSP\\Scalable_Audio_pitch_QDSP\\dsp_out.txt","a+");

	fprintf(fcheck,"%d,%d,%d,%d,%d \n ", (int)xcorr_0,(int)xcorr_1,(int)xcorr_2,(int)xcorr_3,(int)xcorr_4);
	fclose(fcheck);
	*/
	xcorr[0] = (int)xcorr_0;
	xcorr[1] = (int)xcorr_1;
	xcorr[2] = (int)xcorr_2;
	xcorr[3] = (int)xcorr_3;
	xcorr[4] = (int)xcorr_4;
#endif
	return 0;
}