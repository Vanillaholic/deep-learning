#include "config.h"
#include "mathops.h"

//计算_v的对数
int ec_ilog(unsigned int _v){

	int ret;
	int m;
	ret=!!_v;//！！判断_v是否为0
	m=!!(_v&0xFFFF0000)<<4;
	_v>>=m;
	ret|=m;
	m=!!(_v&0xFF00)<<3;
	_v>>=m;
	ret|=m;
	m=!!(_v&0xF0)<<2;
	_v>>=m;
	ret|=m;
	m=!!(_v&0xC)<<1;
	_v>>=m;
	ret|=m;
	ret+=!!(_v&0x2);
	return ret;
}
int frac_div32(int a, int b)
{
 
	short rcp;

   int result, rem;
   int shift = ssc_ilog2(b)-29;
   a = VSHR32(a,shift);
   b = VSHR32(b,shift);
   /* 16-bit reciprocal */
   
   rcp = ROUND16(ssc_rcp(ROUND16(b,16)),3);
   result = MULT16_32_Q15(rcp, a);

   
   rem = PSHR32(a,2)-MULT32_32_Q31(result, b);
   result = ADD32(result, SHL32(MULT16_32_Q15(rcp, rem),2));



   if (result >= 536870912)       /*  2^29 */
      return 2147483647;          /*  2^31 - 1 */
   else if (result <= -536870912) /* -2^29 */
      return -2147483647;         /* -2^31 */
   else
      return SHL32(result, 2);
}

/** Reciprocal sqrt approximation in the range [0.25,1) (Q16 in, Q14 out) */
short ssc_rsqrt_norm(int x)
{
   short n;
   short r;
   short r2;
   short y;
   /* Range of n is [-16384,32767] ([-0.5,1) in Q15). */
   n = x-32768;
   /* Get a rough initial guess for the root.
      The optimal minimax quadratic approximation (using relative error) is
       r = 1.437799046117536+n*(-0.823394375837328+n*0.4096419668459485).
      Coefficients here, and the final result r, are Q14.*/
   r = ADD16(23557, MULT16_16_Q15(n, ADD16(-13490, MULT16_16_Q15(n, 6713))));
   /* We want y = x*r*r-1 in Q15, but x is 32-bit Q16 and r is Q14.
      We can compute the result from n and r using Q15 multiplies with some
       adjustment, carefully done to avoid overflow.
      Range of y is [-1564,1594]. */
   r2 = MULT16_16_Q15(r, r);
   y = SHL16(SUB16(ADD16(MULT16_16_Q15(r2, n), r2), 16384), 1);
   /* Apply a 2nd-order Householder iteration: r += r*y*(y*0.375-0.5).
      This yields the Q14 reciprocal square root of the Q16 x, with a maximum
       relative error of 1.04956E-4, a (relative) RMSE of 2.80979E-5, and a
       peak absolute error of 2.26591/16384. */
   return ADD16(r, MULT16_16_Q15(r, MULT16_16_Q15(y,
              SUB16(MULT16_16_Q15(y, 12288), 16384))));
}

/** Sqrt approximation (QX input, QX/2 output) */
int ssc_sqrt(int x)
{
   int k;
   short n;
   int rt;
   static const short C[5] = {23175, 11561, -3011, 1699, -664};
   if (x==0)
      return 0;
   else if (x>=1073741824)
      return 32767;
   k = (ssc_ilog2(x)>>1)-7;
   x = VSHR32(x, 2*k);
   n = x-32768;
   rt = ADD16(C[0], MULT16_16_Q15(n, ADD16(C[1], MULT16_16_Q15(n, ADD16(C[2],
              MULT16_16_Q15(n, ADD16(C[3], MULT16_16_Q15(n, (C[4])))))))));
   rt = VSHR32(rt,7-k);
   return rt;
}



/** Reciprocal approximation (Q15 input, Q16 output) */
int ssc_rcp(int x)
{
   int i;
   short n;
   short r;
   ssc_assert2(x>0, "ssc_rcp() only defined for positive values");
   i = ssc_ilog2(x);
   /* n is Q15 with range [0,1). */
   n = VSHR32(x,i-15)-32768;
   /* Start with a linear approximation:
      r = 1.8823529411764706-0.9411764705882353*n.
      The coefficients and the result are Q14 in the range [15420,30840].*/
   r = ADD16(30840, MULT16_16_Q15(-15420, n));
   /* Perform two Newton iterations:
      r -= r*((r*n)-1.Q15)
         = r*((r*n)+(r-1.Q15)). */
   r = SUB16(r, MULT16_16_Q15(r,
             ADD16(MULT16_16_Q15(r, n), ADD16(r, -32768))));
   /* We subtract an extra 1 in the second iteration to avoid overflow; it also
       neatly compensates for truncation error in the rest of the process. */
   r = SUB16(r, ADD16(1, MULT16_16_Q15(r,
             ADD16(MULT16_16_Q15(r, n), ADD16(r, -32768)))));
   /* r is now the Q15 solution to 2/(n+1), with a maximum relative error
       of 7.05346E-5, a (relative) RMSE of 2.14418E-5, and a peak absolute
       error of 1.24665/32768. */
   return VSHR32(EXTEND32(r),i-16);
}


