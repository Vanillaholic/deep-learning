

#include "config.h"
#include "ssc.h"
#include "modes.h"



#define BITALLOC_SIZE 11


  #include "static_modes_fixed.h"


#ifndef M_PI
#define M_PI 3.141592653
#endif


SSCMode *ssc_custom_mode_create(int Fs, int frame_size, int *error)
{

	if (error)
		*error = SSC_OK;

#ifndef HW_48kHz
	if(Fs == 44100)
	{
		 return (SSCMode*)static_mode_list[0];
	}
#else
	if(Fs == 44100 || Fs == 48000)
	{
		return (SSCMode*)static_mode_list[0];
	}
#endif
	else if(Fs == 96000)
	{
		 return (SSCMode*)static_mode_list[1];
	}
	else
	{
		if (error)
			*error = SSC_BAD_ARG;
	}


   return NULL;

}

