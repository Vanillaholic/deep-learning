
#ifndef	AUDIO_WAV_H
#define	AUDIO_WAV_H

#include <stdio.h>

typedef	struct	_WAV_HEADER
{
	unsigned char	chunk_id[4];
	unsigned long	chunk_size;
	unsigned char	wav_format[4];
	unsigned char	format_chunk_id[4];
	unsigned long	format_chunk_size;
	unsigned short	audio_format;
	unsigned short	channels;
	unsigned long	sample_rate;
	unsigned long	byte_rate;
	unsigned short	byte_align;
	unsigned short	bits_per_sample;
	unsigned char	data_chunk_id[4];
	unsigned long	data_chunk_size;
}	WAV_HEADER;

typedef	enum	_WAV_MODE
{
	WAV_READ	= 0,
	WAV_WRITE
}	WAV_MODE;

FILE*	wav_open(char* szWavName, WAV_HEADER* pHdr, int mode);
int		wav_close(FILE* fp, WAV_HEADER* pHdr, int mode);


#endif	//	#ifndef	AUDIO_WAV_H
