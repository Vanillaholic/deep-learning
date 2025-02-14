#include "config.h"
#include "pitch.h"
#include "modes.h"
#include "mathops.h"
#include "ssc_lpc.h"
#include <stdlib.h>

#include <stdio.h>

#ifdef HW_QDSP_OPTI
#include "basic_op_qdsp.h"
#endif

#ifdef HW_STACK_OPT
//extern int _pre_shared[3776];   // (1024+864)*2
extern int in_shared[1944];			//CC*(N+st->overlap)
#endif

#ifndef find_best_pitch_QDSP_OPT
static void find_best_pitch(int *xcorr, short *y, int len,
	int max_pitch, int *best_pitch, int yshift, int maxcorr)
{
	int i, j;
	int Syy=1;
	short best_num[2];
	int best_den[2];
	int xshift;
	short num;
	int xcorr16;


	xshift = ssc_ilog2(maxcorr)-14;

	best_num[0] = -1;
	best_num[1] = -1;
	best_den[0] = 0;
	best_den[1] = 0;
	best_pitch[0] = 0;
	best_pitch[1] = 1;



	for (j = len-1; j >= 0;j--)
		Syy = ADD32(Syy, SHR32(MULT16_16(y[j],y[j]), yshift));


	for (i=0;i<max_pitch;i++)
	{
		if (xcorr[i]>0)
		{

			xcorr16 = EXTRACT16(VSHR32(xcorr[i], xshift));
			num = MULT16_16_Q15(xcorr16,xcorr16);

#ifndef HW_QDSP_OPTI
			if (MULT16_32_Q15(num,best_den[1]) > MULT16_32_Q15(best_num[1],Syy))
			{
				if (MULT16_32_Q15(num,best_den[0]) > MULT16_32_Q15(best_num[0],Syy))
				{
#else
			if (MULT16_32_Q15_qdsp(num, best_den[1]) > MULT16_32_Q15_qdsp(best_num[1], Syy))
			{
				if (MULT16_32_Q15_qdsp(num, best_den[0]) > MULT16_32_Q15_qdsp(best_num[0], Syy))
				{
#endif
					best_num[1] = best_num[0];
					best_den[1] = best_den[0];
					best_pitch[1] = best_pitch[0];
					best_num[0] = num;
					best_den[0] = Syy;
					best_pitch[0] = i;
				} 
				else 
				{
					best_num[1] = num;
					best_den[1] = Syy;
					best_pitch[1] = i;
				}
			}
		}
		Syy += SHR32(MULT16_16(y[i+len],y[i+len]),yshift) - SHR32(MULT16_16(y[i],y[i]),yshift);
		Syy = MAX32(1, Syy);
	}
}
#else
static void find_best_pitch(int *xcorr, short *y, int len,
	int max_pitch, int *best_pitch, int yshift, int maxcorr)
{
	int i, j;
	int Syy=1;
	short best_num[2];
	int best_den[2];
	int xshift;
	short num;
	int xcorr16;

	long long *pt;
	long long tmpLL1,tmpLL2, sum;


	xshift = ssc_ilog2(maxcorr)-14;

	best_num[0] = -1;
	best_num[1] = -1;
	best_den[0] = 0;
	best_den[1] = 0;
	best_pitch[0] = 0;
	best_pitch[1] = 1;


	if(yshift == 0)
	{
		pt = (long long *)y;
		sum = 0;
		for (j=0;j<len;j+=8)
		{
			tmpLL1 = *pt++;
			tmpLL2 = *pt++;
			sum = Q6_P_vdmpyacc_PP_sat(sum,tmpLL1,tmpLL1);
			sum = Q6_P_vdmpyacc_PP_sat(sum,tmpLL2,tmpLL2);

		}
		Syy += Q6_R_add_RR((sum&0xffffffff),sum>>32);


		for (i=0;i<max_pitch;i++)
		{
			if (xcorr[i]>0)
			{

				xcorr16 = EXTRACT16(VSHR32(xcorr[i], xshift));
				num = MULT16_16_Q15(xcorr16,xcorr16);

				if (MULT16_32_Q15(num,best_den[1]) > MULT16_32_Q15(best_num[1],Syy))
				{
					if (MULT16_32_Q15(num,best_den[0]) > MULT16_32_Q15(best_num[0],Syy))
					{

						best_num[1] = best_num[0];
						best_den[1] = best_den[0];
						best_pitch[1] = best_pitch[0];
						best_num[0] = num;
						best_den[0] = Syy;
						best_pitch[0] = i;
					} 
					else 
					{
						best_num[1] = num;
						best_den[1] = Syy;
						best_pitch[1] = i;
					}
				}
			}
			Syy += MULT16_16(y[i+len],y[i+len]) - MULT16_16(y[i],y[i]);
			Syy = MAX32(1, Syy);
		}
	}else
	{

		pt = (long long *)y;
		sum = 0;
		for (j=0;j<len;j+=8)
		{
			tmpLL1 = *pt++;
			tmpLL2 = *pt++;
			sum = Q6_P_vdmpyacc_PP_sat(sum,tmpLL1,tmpLL1);
			sum = Q6_P_vdmpyacc_PP_sat(sum,tmpLL2,tmpLL2);

		}
		sum=Q6_P_vasrw_PR(sum,yshift);
		Syy += Q6_R_add_RR((sum&0xffffffff),sum>>32);


		for (i=0;i<max_pitch;i++)
		{
			if (xcorr[i]>0)
			{

				xcorr16 = EXTRACT16(VSHR32(xcorr[i], xshift));
				num = MULT16_16_Q15(xcorr16,xcorr16);


				if (MULT16_32_Q15(num,best_den[1]) > MULT16_32_Q15(best_num[1],Syy))
				{
					if (MULT16_32_Q15(num,best_den[0]) > MULT16_32_Q15(best_num[0],Syy))
					{

						best_num[1] = best_num[0];
						best_den[1] = best_den[0];
						best_pitch[1] = best_pitch[0];
						best_num[0] = num;
						best_den[0] = Syy;
						best_pitch[0] = i;
					} 
					else 
					{
						best_num[1] = num;
						best_den[1] = Syy;
						best_pitch[1] = i;
					}
				}
			}
			Syy += SHR32(MULT16_16(y[i+len],y[i+len]),yshift) - SHR32(MULT16_16(y[i],y[i]),yshift);
			Syy = MAX32(1, Syy);
		}
	}
}
#endif

#ifndef ssc_fir5_QDSP_OPTI
void ssc_fir5(short *x, const short *num, short *y)
{
	int i;
	short num0, num1, num2, num3, num4;
	int mem0, mem1, mem2, mem3, mem4;
	int sum;

	num0=num[0];
	num1=num[1];
	num2=num[2];
	num3=num[3];
	num4=num[4];

	mem1 = mem2 = mem3 = mem4 = 0;
	mem0 = x[0];
	sum = SHL32(EXTEND32(x[0]), SIG_SHIFT);
	y[0] = ROUND16(sum, SIG_SHIFT);

	for (i=1;i<944;i++)
	{
		sum = SHL32(EXTEND32(x[i]), SIG_SHIFT);
		sum = MAC16_16(sum,num0,mem0);
		sum = MAC16_16(sum,num1,mem1);
		sum = MAC16_16(sum,num2,mem2);
		sum = MAC16_16(sum,num3,mem3);
		sum = MAC16_16(sum,num4,mem4);
		mem4 = mem3;
		mem3 = mem2;
		mem2 = mem1;
		mem1 = mem0;
		mem0 = x[i];
		y[i] = ROUND16(sum, SIG_SHIFT);
	}

}
#else
void ssc_fir5( short *x, const short *num, short *y)
{
	int i;

	long long coef0,coef1, *pt64, *ptx, mem0, mem1, *pty;
	long long sum0, sum1, sum2, sum3;
	short mem[8] = {0};
	short coeff[8] = {0};


	coeff[3] = 4096;
	coeff[2]=num[0];
	coeff[1]=num[1];
	coeff[0]=num[2];

	coeff[7]=num[3];
	coeff[6]=num[4];

	pt64 = (long long *)coeff;
	coef0 = *pt64++;
	coef1 = *pt64;
	pt64 = (long long*)mem;
	mem0 = *pt64++;
    mem1 = *pt64;


	ptx = (long long*)x;
	pty = (long long*)y;
	for (i=0;i<944;i+=4)
	{
		long long xv, tmpll0, tmpll1;
		xv = *ptx++;
		tmpll0 = ((mem0 >> 16)&0xffffffffffff)|((xv&0xffff)<<48);
		tmpll1 = ((mem1 >> 16)&0xffffffffffff)|((mem0&0xffff)<<48);
//		tmpll0 = Q6_P_valignb_PPI(xv,mem0,2);
//		tmpll1 = Q6_P_valignb_PPI(mem0,mem1,6);
		sum0 = Q6_P_vrmpyh_PP(tmpll0,coef0);
        sum0 = Q6_P_vrmpyhacc_PP(sum0,tmpll1,coef1);
		sum0 = Q6_P_asr_PI_rnd(sum0, SIG_SHIFT-1);



		tmpll0 = ((mem0 >> 32)&0xffffffff)|((xv&0xffffffff)<<32);
     	tmpll1 = ((mem1 >> 32)&0xffffffff)|((mem0&0xffffffff)<<32);
//		tmpll0 = Q6_P_valignb_PPI(mem0,xv,4);
//		tmpll1 = Q6_P_valignb_PPI(mem0,mem1,4);
		sum1 = Q6_P_vrmpyh_PP(tmpll0,coef0);
	    sum1 = Q6_P_vrmpyhacc_PP(sum1,tmpll1,coef1);
		sum1 = Q6_P_asr_PI_rnd(sum1, SIG_SHIFT-1);


	    tmpll0 = ((mem0 >> 48)&0xffff)|((xv&0xffffffffffff)<<16);
		tmpll1 = ((mem1 >> 48)&0xffff)|((mem0&0xffffffffffff)<<16);

//		tmpll0 = Q6_P_valignb_PPI(mem0,xv,6);
//		tmpll1 = Q6_P_valignb_PPI(mem0,mem1,6);
		sum2 = Q6_P_vrmpyh_PP(tmpll0,coef0);
	    sum2 = Q6_P_vrmpyhacc_PP(sum2,tmpll1,coef1);
		sum2 = Q6_P_asr_PI_rnd(sum2, SIG_SHIFT-1);

		sum3 = Q6_P_vrmpyh_PP(xv,coef0);
	    sum3 = Q6_P_vrmpyhacc_PP(sum3,mem0,coef1);
		sum3 = Q6_P_asr_PI_rnd(sum3, SIG_SHIFT-1);


		mem1 = mem0;
		mem0 = xv;

	    tmpll0 = Q6_P_combine_RR(Q6_R_combine_RlRl(sum3, sum2), Q6_R_combine_RlRl(sum1, sum0));
	    *pty++ = tmpll0;
		
	}



}
#endif


FILE *fp;

void pitch_downsample(int *x_0,int *x_1, short *x_lp, int len)
{
	int i;
	int ac[5];
	short tmp=Q15ONE;
	short lpc[4]; 

	short lpc2[5];
	short c1 = 26214;

	int shift;

	int maxabs = ssc_maxabs32(x_0, len<<1);


	if (maxabs<1)
		maxabs=1;

#ifndef QDSP_INTRINSIC
	shift = ssc_ilog2(maxabs)-10;
#else
	shift = 21 - __builtin_clz(maxabs);
#endif
	if (shift<0)
		shift=0;

	shift++;
	

#ifndef  pitch_downsample_QDSP_OPT
	x_lp[0] = SHR32(HALF32(HALF32(x_0[1])+x_0[0]), shift);
	x_lp[0] += SHR32(HALF32(HALF32(x_1[1])+x_1[0]), shift);

/*	for (i=1;i<len>>1;i++)
	{
		x_0[i] = i*40000; 
		x_1[i] = i*80000; 
	}
	*/
	for (i=1;i<len>>1;i++)
	{
 		x_lp[i] = SHR32(HALF32(HALF32(x_0[((i<<1)-1)]+x_0[((i<<1)+1)])+x_0[i<<1]), shift);
		x_lp[i] += SHR32(HALF32(HALF32(x_1[((i<<1)-1)]+x_1[((i<<1)+1)])+x_1[i<<1]), shift);
	}
		fp = fopen("E:\\2019_QDSP\\Scalable_Audio_pitch_QDSP\\test_vectors\\c.txt","a+");
		for (i=1;i<len>>1;i++)
		{
			fprintf(fp,"%d\n", x_lp[i]);
		}
		fclose(fp);

#else
	{
		long long *pt0,*pt1, dx0_0, dx0_1, dx0_2,dx1_0,dx1_1,dx1_2, tmpll;
		int shift1;
		x_lp[0] = SHR32(HALF32(HALF32(x_0[1])+x_0[0]), shift);
		x_lp[0] += SHR32(HALF32(HALF32(x_1[1])+x_1[0]), shift);


		
		pt0 = (long long *)&x_0[0];
		pt1 = (long long *)&x_1[0];
		dx0_0 = *pt0++;  
		dx1_0 = *pt1++;  
		shift1 = shift +2;
		for (i=1;i<len>>1;i++)
		{
			int tmpl, tmpl1;
			dx0_1 = *pt0++;  
			dx0_2 = (dx0_0&0xffffffff00000000)|(dx0_1&0x00000000ffffffff);

			tmpll = Q6_P_vaddw_PP(dx0_1, dx0_2);
			tmpl =  Q6_R_add_RR((tmpll&0xffffffff),tmpll>>32);
		    tmpl1 =  tmpl >> shift1;

			dx1_1 = *pt1++;  
			dx1_2 = (dx1_0&0xffffffff00000000)|(dx1_1&0x00000000ffffffff);
			tmpll = Q6_P_vaddw_PP(dx1_1, dx1_2);
	     	tmpl =  Q6_R_add_RR((tmpll&0xffffffff),tmpll>>32);
		    x_lp[i] =  tmpl1 + (tmpl >> shift1);
	

			dx0_0 = dx0_1;
			dx1_0 = dx1_1;
		}
		
	/*	fp = fopen("E:\\2019_QDSP\\Scalable_Audio_pitch_QDSP\\d.txt","a+");
		for (i=1;i<len>>1;i++)
		{
			fprintf(fp,"%d\n", x_lp[i]);
		}
		fclose(fp);*/

	}
#endif





	_ssc_autocorr(x_lp, ac);   // (len>>1) = 944

	/* Noise floor -40 dB */
	ac[0] += SHR32(ac[0],13);

	/* Lag windowing */	

	ac[1] -= (ac[1] >> 14);	//MULT16_32_Q15(2,  ac[1]);
	ac[2] -= (ac[2] >> 12);	// MULT16_32_Q15(8,  ac[2]);	//
#ifndef HW_QDSP_OPTI
	ac[3] -= MULT16_32_Q15(18, ac[3]);	// ((ac[3] * 18) >> 15);	//
#else
	ac[3] -= MULT16_32_Q15_qdsp(18, ac[3]);	// ((ac[3] * 18) >> 15);	//
#endif
	ac[4] -= (ac[4] >> 10);	// MULT16_32_Q15(32, ac[4]);	// 
	



	_ssc_lpc(lpc, ac);


	{
		int tmp_macro = 29491;

		tmp = MULT16_16_Q15(tmp_macro, tmp);
		lpc[0] = MULT16_16_Q15(lpc[0], tmp);

		tmp = MULT16_16_Q15(tmp_macro, tmp);
		lpc[1] = MULT16_16_Q15(lpc[1], tmp);

		tmp = MULT16_16_Q15(tmp_macro, tmp);
		lpc[2] = MULT16_16_Q15(lpc[2], tmp);

		tmp = MULT16_16_Q15(tmp_macro, tmp);
		lpc[3] = MULT16_16_Q15(lpc[3], tmp);
	}

	/* Add a zero */
	lpc2[0] = lpc[0] + 3277;
	lpc2[1] = lpc[1] + MULT16_16_Q15(c1,lpc[0]);
	lpc2[2] = lpc[2] + MULT16_16_Q15(c1,lpc[1]);
	lpc2[3] = lpc[3] + MULT16_16_Q15(c1,lpc[2]);
	lpc2[4] = MULT16_16_Q15(c1,lpc[3]);

	ssc_fir5(x_lp, lpc2, x_lp);

}



void pitch_search(const short *x_lp, short *y, int len, int max_pitch, int *pitch)
{
	int i, j;
	int lag;
	int best_pitch[2]={0,0};

#ifndef HW_STACK_OPT
	short x_lp4[216];	//len>>2
	short y_lp4[484];	//lag>>2
	int xcorr[489];	//max_pitch>>1
#else
#ifndef ssc_pitch_xcorr_c2_QDSP_OPTI
	short *x_lp4 = (short *)(&in_shared[477]);	//944/2 = 477
	short *y_lp4 = (short *)(&in_shared[585]);	//477+108
	int *xcorr = (int *)(&in_shared[827]);   // 585+242
#else
	short *x_lp4 = (short *)(&in_shared[480]);	//944/2 = 477
	short *y_lp4 = (short *)(&in_shared[588]);	//477+108
	int *xcorr = (int *)(&in_shared[832]);   // 585+242
#endif
#endif


	int maxcorr;
	
	
	int offset;

	int sum;
	int a, b, c;



	lag = len+max_pitch;


#ifndef pitch_search_loop1_QDSQ_OPT  
    for (j=0;j<len>>2;j++)
    {
        x_lp4[j] = x_lp[j<<1];
        y_lp4[j] = y[j<<1];
    }
    for (;j<lag>>2;j++)
        y_lp4[j] = y[j<<1];
#else
    long long tmp1, tmp2, tmp3, tmp4;

    long long *pt1=(long long*)x_lp;
    long long *pt2=(long long*)y;
    long long *pt1_4=(long long*)x_lp4;
    long long *pt2_4=(long long*)y_lp4;

    for (j=0;j<len>>2;j+=8)
    {
        tmp1=*pt1++;
        tmp2=*pt1++;
		tmp3=*pt1++;
        tmp4=*pt1++;
        *pt1_4++=Q6_P_vtrunewh_PP(tmp2,tmp1);//
        *pt1_4++=Q6_P_vtrunewh_PP(tmp4,tmp3);//     
    }

    for (j=0;j<lag>>2;j+=4)
    {
        tmp1=*pt2++;
        tmp2=*pt2++;
        *pt2_4++=Q6_P_vtrunewh_PP(tmp2,tmp1);
        //*pt2_4++=tmp1;
    }
#endif


	/* Coarse search with 4x decimation */
	maxcorr = ssc_pitch_xcorr_c2(x_lp4, y_lp4, xcorr, len>>2, max_pitch>>2);


	find_best_pitch(xcorr, y_lp4, len>>2, max_pitch>>2, best_pitch, 0, maxcorr);

	/* Finer search with 2x decimation */
	maxcorr=1;


	for (i=0;i<max_pitch>>1;i++)
	{
		sum = 0;
		xcorr[i] = 0;
		if (abs(i - 2*best_pitch[0]) > 2 && abs(i - 2*best_pitch[1])>2)
			continue;

#ifndef pitch_search_loop2_QDSP_OPT
        for (j=0;j<len>>1;j++) sum += MULT16_16(x_lp[j],y[i+j]);
#else
        {
            char s;
            int alg;
            long long d0, d1, d2, d3;
            long long *p0, *p1;
            
            d3 = 0;
            alg = (4-i&0x00000003);
            for (j=0;j<alg;j++) d3 += MULT16_16(x_lp[j],y[i+j]);
            s = (char)(alg<<1);
            p0 = (long long *)&x_lp[0];
            p1 = (long long *)&y[i+alg];
            d0 = p0[0];
            d1 = p0[1];
            d2 = p1[0];
            for(j=0; j<(len>>3)-1; j++)
            {
                long long d;
                d = Q6_P_valignb_PPp(d1, d0, s);
                d0 = d1;
                d1 = p0[j+2];
                d3 = Q6_P_vrmpyhacc_PP(d3, d, d2);
                d2 = p1[j+1];
            }
            for (j=(len>>1)-(4-alg);j<(len>>1);j++) d3 += MULT16_16(x_lp[j],y[i+j]);
            sum = (int)d3;
        }
#endif




		xcorr[i] = MAX32(-1, sum);
		maxcorr = MAX32(maxcorr, sum);
	}


	find_best_pitch(xcorr, y, len>>1, max_pitch>>1, best_pitch, 1, maxcorr);


	/* Refine by pseudo-interpolation */
	if (best_pitch[0]>0 && best_pitch[0]<(max_pitch>>1)-1)
	{	
		a = xcorr[best_pitch[0]-1];
		b = xcorr[best_pitch[0]];
		c = xcorr[best_pitch[0]+1];
#ifndef HW_QDSP_OPTI
		if ((c-a) > MULT16_32_Q15(22938,b-a))
			offset = 1;
		else if ((a-c) > MULT16_32_Q15(22938,b-c))
			offset = -1;
		else
			offset = 0;
#else
		if ((c - a) > MULT16_32_Q15_qdsp(22938, b - a))
			offset = 1;
		else if ((a - c) > MULT16_32_Q15_qdsp(22938, b - c))
			offset = -1;
		else
			offset = 0;
#endif
	} 
	else 
	{
		offset = 0;
	}
	*pitch = 2*best_pitch[0]-offset;
}

static const int second_check[16] = {0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2};
short remove_doubling(short *x, int maxperiod, int minperiod,
	int N, int *T0_, int prev_period, short prev_gain)
{
	int k, i, T, T0;
	short g, g0;
	short pg;
	int xy,xx,yy,xy2;

	int xcorr[3] = {0,};

	int best_xy, best_yy;
	int offset;
	int minperiod0;

#ifndef HW_STACK_OPT
	int yy_lookup[513];
#else
	int *yy_lookup = (int *)(&in_shared[477]);
#endif
	int T1, T1b;
	short g1;
	short cont=0;
	short thresh;
	int x2y2;
	int sh, t;
//	int T1;

	minperiod0 = minperiod;

	maxperiod >>=1;
	minperiod >>=1;
	*T0_ >>=1;
	prev_period >>=1;
	N >>=1;

	x += maxperiod;
	if (*T0_>=maxperiod)
		*T0_=maxperiod-1;

	T = T0 = *T0_;
	dual_inner_prod(x, x, x-T0, N, &xx, &xy);
	yy_lookup[0] = xx;
	yy=xx;
#ifndef remove_doubling_loop1_QDSP_OPT
	for (i=1;i<=maxperiod;i++)
	{
		yy = yy+MULT16_16(x[-i],x[-i])-MULT16_16(x[N-i],x[N-i]);	

		yy_lookup[i] = yy;

	}
#else
	{

		int *pt1, *pt2;
		pt1 = (int*)&x[-2];
		pt2 = (int*)&x[N-2];
		for (i=1;i<=maxperiod;i+=2)
		{
			int d1, d2, tmp;
			d1 = *pt1--;
			d2 = *pt2--;
			tmp = Q6_R_mpyacc_RhRh(yy, d1, d1) ;
			yy = Q6_R_mpynac_RhRh(tmp,d2, d2);
            yy_lookup[i] = yy;
			tmp = Q6_R_mpyacc_RlRl(yy, d1, d1) ;
			yy = Q6_R_mpynac_RlRl(tmp,d2, d2);
            yy_lookup[i+1] = yy;
		}

	}
#endif
	yy = yy_lookup[T0];
	best_xy = xy;
	best_yy = yy;
	{
		x2y2 = 1+HALF32(MULT32_32_Q31(xx,yy));
		sh = ssc_ilog2(x2y2)>>1;
		t = VSHR32(x2y2, 2*(sh-7));
		g = g0 = VSHR32(MULT16_32_Q15(ssc_rsqrt_norm(t), xy),sh+1);
	}
	/* Look for any pitch at T/k */
	for (k=2;k<=15;k++)
	{

		T1 = (2*T0+k)/(2*k);
		if (T1 < minperiod)
			break;
		/* Look for another strong correlation at T1b */
		if (k==2)
		{
			if (T1+T0>maxperiod)
				T1b = T0;
			else
				T1b = T0+T1;
		} 
		else
		{
			T1b = (2*second_check[k]*T0+k)/(2*k);
		}
		dual_inner_prod(x, &x[-T1], &x[-T1b], N, &xy, &xy2);
		xy += xy2;
		yy = yy_lookup[T1] + yy_lookup[T1b];

		x2y2 = 1+MULT32_32_Q31(xx,yy);
		sh = ssc_ilog2(x2y2)>>1;
		t = VSHR32(x2y2, 2*(sh-7));
#ifndef HW_QDSP_OPTI
		g1 = VSHR32(MULT16_32_Q15(ssc_rsqrt_norm(t), xy),sh+1);
#else
		g1 = VSHR32(MULT16_32_Q15_qdsp(ssc_rsqrt_norm(t), xy), sh + 1);
#endif
		if (abs(T1-prev_period)<=1)
			cont = prev_gain;
		else if (abs(T1-prev_period)<=2 && 5*k*k < T0)
			cont = HALF32(prev_gain);
		else
			cont = 0;
		thresh = MAX16(9830, MULT16_16_Q15(22938,g0)-cont);
		/* Bias against very high pitch (very short period) to avoid false-positives
		due to short-term correlation */
		if (T1<3*minperiod)
			thresh = MAX16(13107, MULT16_16_Q15(27853,g0)-cont);
		else if (T1<2*minperiod)
			thresh = MAX16(16384, MULT16_16_Q15(29491,g0)-cont);
		if (g1 > thresh)
		{
			best_xy = xy;
			best_yy = yy;
			T = T1;
			g = g1;
		}
	}
	best_xy = MAX32(0, best_xy);
	if (best_yy <= best_xy)
		pg = Q15ONE;
	else
		pg = SHR32(frac_div32(best_xy,best_yy+1),16);


#ifndef remove_doubling_loop2_QDSP_OPT
	for (i=0;i<N;i++)
	{
		xcorr[0] = MAC16_16(xcorr[0], x[i], x[i-(T-1)]);
		xcorr[1] = MAC16_16(xcorr[1], x[i], x[i-T]);
		xcorr[2] = MAC16_16(xcorr[2], x[i], x[i-(T+1)]);
	}
#else
	if((T & 0x1) == 1)
	{
		int *pt1, *pt2;
		int x0, xd0, xd1;
		pt1 = (int *)&x[0];
		pt2 = (int *)&x[-(T+1)];

		xd0 = *pt2++;
		for (i=0;i<N;i+=2)
		{
			x0  = *pt1++;
			xd1 = *pt2++;

			xcorr[0] = Q6_R_mpyacc_RlRl(xcorr[0],x0,xd1);
			xcorr[1] = Q6_R_mpyacc_RlRh(xcorr[1],x0,xd0);
			xcorr[2] = Q6_R_mpyacc_RlRl(xcorr[2],x0,xd0);

		    xcorr[0] = Q6_R_mpyacc_RhRh(xcorr[0],x0,xd1);
			xcorr[1] = Q6_R_mpyacc_RhRl(xcorr[1],x0,xd1);
			xcorr[2] = Q6_R_mpyacc_RhRh(xcorr[2],x0,xd0);

			xd0 = xd1;


		}
	}else
	{
		for (i=0;i<N;i++)
		{
			xcorr[0] = MAC16_16(xcorr[0], x[i], x[i-(T-1)]);
			xcorr[1] = MAC16_16(xcorr[1], x[i], x[i-T]);
			xcorr[2] = MAC16_16(xcorr[2], x[i], x[i-(T+1)]);
		}

	}
#endif

#ifndef HW_QDSP_OPTI
	if ((xcorr[2]-xcorr[0]) > MULT16_32_Q15(22938,xcorr[1]-xcorr[0]))
		offset = 1;
	else if ((xcorr[0]-xcorr[2]) > MULT16_32_Q15(22938,xcorr[1]-xcorr[2]))
		offset = -1;
	else
		offset = 0;
#else
	if ((xcorr[2] - xcorr[0]) > MULT16_32_Q15_qdsp(22938, xcorr[1] - xcorr[0]))
		offset = 1;
	else if ((xcorr[0] - xcorr[2]) > MULT16_32_Q15_qdsp(22938, xcorr[1] - xcorr[2]))
		offset = -1;
	else
		offset = 0;
#endif
	if (pg > g)
		pg = g;
	*T0_ = 2*T+offset;

	if (*T0_<minperiod0)
		*T0_=minperiod0;

	return pg;
}

