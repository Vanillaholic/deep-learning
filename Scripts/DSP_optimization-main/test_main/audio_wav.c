
#include <stdio.h>
#include <string.h>
#include "audio_wav.h"


FILE*	wav_open(char* szWavName, WAV_HEADER* pHdr, int mode)
{
	FILE*	fp = NULL;

	if(mode == WAV_READ)
	{
//		fopen_s(&fp, szWavName, "rb");
		fp = fopen(szWavName, "rb");
		if(fp)
		{
			int read_size = fread(pHdr, 1, sizeof(WAV_HEADER), fp);
  //          int flag;
			if(read_size != sizeof(WAV_HEADER))
			{
				return	NULL;
			}

			//	for 18/40 byte check : Audition CS6
//			flag = strncmp((char*)pHdr->data_chunk_id, "data", 4);
			if(strncmp((char*)pHdr->data_chunk_id, "data", 4) &&
				pHdr->data_chunk_id[2] == 'd' &&
				pHdr->data_chunk_id[3] == 'a')
			{
				fseek(fp, 2, SEEK_CUR);
			}
		}
	}
	else if (mode == WAV_WRITE)
	{
//		fopen_s(&fp, szWavName, "wb");
		fp = fopen(szWavName, "wb");
		if(fp)
		{
			fseek(fp, sizeof(WAV_HEADER), SEEK_SET);
		}
	}
	

	return	fp;
}

int	wav_close(FILE* fp, WAV_HEADER* pHdr, int mode)
{
	int		error = 1;

	if(mode == WAV_READ)
	{
		fclose(fp);
		error = 0;
	}
	else if(mode == WAV_WRITE)
	{
		int		data_size;

		fseek(fp, 0, SEEK_END);
		data_size = ftell(fp) - sizeof(WAV_HEADER);
		fseek(fp, 0, SEEK_SET);

		memcpy(pHdr->chunk_id,        "RIFF", 4);		
		pHdr->chunk_size		= sizeof(WAV_HEADER) - 8;		//	8 : chunk ID (4) + size field (4)
		memcpy(pHdr->wav_format,      "WAVE", 4);
		memcpy(pHdr->format_chunk_id, "fmt ", 4);
		pHdr->format_chunk_size	= 16;
		pHdr->audio_format		= 1;
		//pHdr->channels			= -1;
		//pHdr->sample_rate		= -1;
		pHdr->byte_rate		= pHdr->sample_rate * pHdr->channels * pHdr->bits_per_sample / 8;
		pHdr->byte_align	= pHdr->channels * pHdr->bits_per_sample / 8;
		//pHdr->bits_per_sample	= -1;
		memcpy(pHdr->data_chunk_id,    "data", 4);
		pHdr->data_chunk_size	= data_size;

		fwrite(pHdr, 1, sizeof(WAV_HEADER), fp);
		fclose(fp);

		error = 0;
	}	

	return	error;
}