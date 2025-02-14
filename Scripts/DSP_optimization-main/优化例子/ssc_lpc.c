

#include "config.h"
#include "ssc_lpc.h"
#include "mathops.h"
#include "pitch.h"




#ifdef HW_STACK_OPT
extern int in_shared[1944];			//CC*(N+st->overlap)
#endif


void _ssc_lpc(short *_lpc, int *ac)
{
   int i, j;
   int r,rr;
   int ac0 = *ac++;
   int error = ac0;

   int lpc[4] = {0,};


   if (ac0 != 0)
   {

      i=1;
	
	  rr = SHR32(*ac,3);
      r = -frac_div32(SHL32(rr,3), error);
		   /*  Update LPC coefficients and total error */
      lpc[0] = SHR32(r,3);

	  error = error - MULT32_32_Q31(MULT32_32_Q31(r,r),error);
		   /* Bail out once we get 30 dB gain */
	  if (error<SHR32(ac0,10))
		 i=4;

	   
	   for (; i < 4; i++) 
	   {
		   /* Sum up this iteration's reflection coefficient */
		   rr = MULT32_32_Q31(*lpc,*ac);
		   for (j = 1; j < i; j++)
			   rr += MULT32_32_Q31(lpc[j],ac[-j]);

		   rr += SHR32(*(++ac),3);

		   r = -frac_div32(SHL32(rr,3), error);
		   /*  Update LPC coefficients and total error */
		   lpc[i] = SHR32(r,3);
		   for (j = 0; j < (i+1)>>1; j++)
		   {
			   int tmp1, tmp2;
			   tmp1 = lpc[j];
			   tmp2 = lpc[i-1-j];
			   lpc[j]     = tmp1 + MULT32_32_Q31(r,tmp2);
			   lpc[i-1-j] = tmp2 + MULT32_32_Q31(r,tmp1);
		   }

		   error = error - MULT32_32_Q31(MULT32_32_Q31(r,r),error);
		   /* Bail out once we get 30 dB gain */
		   if (error<SHR32(ac0,10))
			   break;
	   }

   }


   _lpc[0] = ROUND16(lpc[0],16);
   _lpc[1] = ROUND16(lpc[1],16);
   _lpc[2] = ROUND16(lpc[2],16);
   _lpc[3] = ROUND16(lpc[3],16);

}


int _ssc_autocorr(const short *x,int *ac)
{
   int d;
   int i;

   int n = 944;


   int shift;
   const short *xptr;

//   short xx[752];

#ifndef HW_STACK_OPT
   short xx[944];
#else
   short *xx = (short *)in_shared;
#endif
   
#ifdef _ssc_autocorr_QDSP_OPTI
   long long *ptx;
   long long sum;
#endif
   
   xptr = x;
   shift=0;

   {
      int ac0;

	  ac0 = 120833;

#ifndef _ssc_autocorr_QDSP_OPTI
	  for(i=0;i<n;i+=2)
	  {
		  ac0 += SHR32(MULT16_16(*xptr,*xptr),9);
		  xptr++;
		  ac0 += SHR32(MULT16_16(*xptr,*xptr),9);
		  xptr++;
	  }
#else
     ptx = (long long *)x;
	 sum = 0;
	 for(i=0;i<n;i+=16)
	 {
		 long long tmpLL,tmpLL1;
		 tmpLL = *ptx++;
		 tmpLL1 = *ptx++;
		 sum = Q6_P_vrmpyhacc_PP(sum, tmpLL, tmpLL);
		 sum = Q6_P_vrmpyhacc_PP(sum, tmpLL1, tmpLL1);
		 tmpLL = *ptx++;
		 tmpLL1 = *ptx++;
		 sum = Q6_P_vrmpyhacc_PP(sum, tmpLL, tmpLL);
		 sum = Q6_P_vrmpyhacc_PP(sum, tmpLL1, tmpLL1);
	 }
	 ac0 = SHR32(sum, 9);
#endif
	  xptr = x;

	  shift = ssc_ilog2(ac0)-20;

	  shift = (shift>>1);


      if (shift>0)
      {

         for(i=0;i<n;i++)
            xx[i] = PSHR32(xptr[i], shift);

         xptr = xx;
      } 
	  else
         shift = 0;
   }

   ssc_pitch_xcorr_c1(xptr, xptr, ac);

   d = 0;
   d = MAC16_16(d, xptr[940], xptr[940]);
   d = MAC16_16(d, xptr[941], xptr[941]);
   d = MAC16_16(d, xptr[942], xptr[942]);
   d = MAC16_16(d, xptr[943], xptr[943]);
   ac[0] += d;

   d = 0;
   d = MAC16_16(d, xptr[941], xptr[940]);
   d = MAC16_16(d, xptr[942], xptr[941]);
   d = MAC16_16(d, xptr[943], xptr[942]);
   ac[1] += d;

   d = 0;
   d = MAC16_16(d, xptr[942], xptr[940]);
   d = MAC16_16(d, xptr[943], xptr[941]);
   ac[2] += d;

   d = MULT16_16(xptr[943], xptr[940]);
   ac[3] += d;



   shift = 2*shift;
   if (shift<=0)
      ac[0] += SHL32((int)1, -shift);
   if (ac[0] < 268435456)
   {
#ifndef HW_CLZ 
	   int shift2 = 29 - EC_ILOG(ac[0]);
#else
	   int shift2;
	   clz32(shift2,ac[0]);
	   shift2 = shift2-3;
#endif
      for (i=0;i<=4;i++)
         ac[i] = SHL32(ac[i], shift2);
      shift -= shift2;
   } else if (ac[0] >= 536870912)
   {
      int shift2=1;
      if (ac[0] >= 1073741824)
         shift2++;
      for (i=0;i<=4;i++)
         ac[i] = SHR32(ac[i], shift2);
      shift += shift2;
   }



   return shift;
}
