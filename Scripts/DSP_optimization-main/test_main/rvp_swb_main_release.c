#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_wav.h"
#include "rvp_swb_api.h"
#include "rvp_swb_config.h"


#ifndef VC_PROJ_FOR_ARM
#include "scs.h"
#include "timer.h"


#include "barman.h"
#define VectorTableOffsetRegister 0xE000ED08
extern unsigned int Image$$VECTORS$$Base;
extern unsigned int Image$$PROCESS_STACK$$ZI$$Limit;


/* Enable the FPU if required */
#ifdef __ARM_FP
extern void $Super$$__rt_lib_init(void);

void $Sub$$__rt_lib_init(void)
{
    /* Enable the FPU in both privileged and user modes by setting bits 20-23 to enable CP10 and CP11 */
    SCS.CPACR = SCS.CPACR | (0xF << 20);
    $Super$$__rt_lib_init();
}
#endif
#endif

/////////////////// global variable for decoder /////////////////
	unsigned char out_pcm[320*3]; // 32kHz 10ms (320sample) ����,  320*2*3 == 5760byte)
	unsigned char dec_handle[9200];  // data alignment issue. (not resolved)
//	int dec_handle[2300];  //

//////////////////////////  global variable for encoder ///////////////////////////////
	int in[320]; // // extended 24bit Mono
	unsigned char in_buf[320* 3]; // packed 24bit Mono
	unsigned char data[100]; // output
	unsigned char enc_handle[3600];
///////////////////////////////////////////////////////////
	char	*argv_t[10];
	char	argv0[] = "HQ_BLE_Enc_Test.exe";

	char   argv1[10];
	char	argv2[320];
	char	argv3[320];
	char    argv4[50];
	char    argv5[10];
	char    argv6[10];
	char    argv7[10];
	char    argv8[10];
	char    argv9[10];
///////////////////////////////////////////////////////////////
	int count;

void to_24bit_val(int *out, unsigned char *in_buf, int size, int uhq_flag)
{
	int i, j;

	for (i = 0; i < size; i++)
	{
		j = (2 + uhq_flag)*i;
		out[i] = 0;

		if (uhq_flag)
		{

			int tmp = (int)in_buf[j + 2];
			int tmp1 = (int)in_buf[j + 1];
			int tmp2 = (int)in_buf[j];
			if (((tmp >> 7) & 1))
			{
				out[i] = (tmp << 16) | out[i] + ((tmp1 << 8)&(65280)) | ((tmp2)&(255)) | 0xff000000;
			}
			else
			{
				out[i] = (tmp << 16) | out[i] + ((tmp1 << 8)&(65280)) | ((tmp2)&(255));
			}
		}
		else
		{
			int tmp = ((int)(in_buf[j + 1]) << 24) >> 24;
			int tmp1 = ((int)(in_buf[j]) << 24) >> 24;
			if (((tmp >> 7) & 1))
			{
				out[i] = (tmp << 8) | (tmp1 & (int)255) | (int)0xffff0000;
			}
			else
			{
				out[i] = (tmp << 8) | (tmp1 & (int)255);
			}
		}
	}
}


int main_enc(int argc, char* argv[]);
int main_dec(int argc, char* argv[]);

#if 0   // for .exe pc version
int \int argc, char* argv[])
{
	if (!strcmp(argv[1], "-e"))
	{
		count = 0;
		main_enc(argc, argv);
	}
	else if (!strcmp(argv[1], "-d"))
	{
		count = 0;
		main_dec(argc, argv);
	}
	else
	{
		printf("Input parameter is not valid\n");
	}

	retrun EXIT_SUCCESS;
}
#else

int main(void)  // for debugging
{
	int		argc_t = 8;

//////////////////////////////////////////////////////////////  
	char    argv1[] = "-e";   // encoder : -e, decoder : -d
//////////////////////////////////////////////////////////////

	argv_t[0]	= argv0;
	argv_t[1]	= argv1;
	argv_t[2]	= argv2;
	argv_t[3] = argv3;
	argv_t[4] = argv4;
	argv_t[5] = argv5;
	argv_t[6]	= argv6;
	argv_t[7] = argv7;

	if (!strcmp(argv1, "-e"))
	{
//////////////////// 32kHz 10ms and 7.5ms///////////////////////////////////////////
		count = 0;
#if 1  
		strcpy(argv2, "D:\\Projects\\RVP_SWB-main\\test_vector\\concat_swb_speech.wav");
		strcpy(argv3, "D:\\Projects\\RVP_SWB-main\\test_vector\\concat_swb_speech_qdsp.sec");

	//	strcpy(argv2, "E:\\2024\\SWB_RVP\\test_vector\\id4_orig_32kHz_24bit_part_mono.wav");
	//	strcpy(argv3, "E:\\2024\\SWB_RVP\\test_vector\\id4_orig_32kHz_24bit_part_mono.sec");

		strcpy(argv4, "1");    // channel 
		strcpy(argv5, "31500");   // 	29byte output, 31.5kbps
		strcpy(argv6, "0");    // header_option, 0 means no header info
		strcpy(argv7, "75");    // 10ms : 100,  7.5ms : 75       // 7.5ms case
#endif
///////////////////// 16kHz 10ms and 7.5ms and RVP mode ///////////////////////////////////////////////////
#if 0
		strcpy(argv2, "..\\..\\..\\test_vector\\km05_mus15_part.wav");
		strcpy(argv3, "..\\..\\..\\test_vector\\km05_mus15_part_1_75.sec");
		strcpy(argv4, "1");    // chmode
//////////////// 16kHz 10ms case /////////////////////////////
		strcpy(argv5, "32000");   // 	bitrate
		strcpy(argv6, "2");    // header_option     , 10ms and 7.5ms case 
		strcpy(argv7, "100");    // 10ms : 100,  7.5ms : 75     , 10ms case  
///////////////////// 16kHz 7.5ms RVP case /////////////////////////	
		strcpy(argv5, "28000");   // 	bitrate		
		strcpy(argv6, "3");    // header_option  ,RVP Mode 		
		strcpy(argv7, "75");    // 10ms : 100,  7.5ms : 75      	
#endif	
//////////////////////////////////////////////////////////////////
		main_enc(argc_t, argv_t);
	}
	else if (!strcmp(argv1, "-d"))
	{
		count = 0;
		argc_t = 10;
		argv_t[8] = argv8;
		argv_t[9] = argv9;
//////////////////// 32kHz 10ms and 7.5ms///////////////////////////////////////////
#if 1
		strcpy(argv2, "D:\\Projects\\RVP_SWB-main\\test_vector\\concat_swb_speech_qdsp.sec");
		strcpy(argv3, "D:\\Projects\\RVP_SWB-main\\test_vector\\concat_swb_speech_32kHz_75_no_header_16bit_dec.wav");

	//	strcpy(argv2, "E:\\2024\\SWB_RVP\\test_vector\\id4_orig_32kHz_24bit_part_mono.sec");
	//	strcpy(argv3, "E:\\2024\\SWB_RVP\\test_vector\\id4_orig_32kHz_24bit_part_mono_ds_dec.wav");

		strcpy(argv4, "1");    // chmode
		strcpy(argv5, "0");    //  bit-depth   0 : 16bit, 1:24bit
		strcpy(argv6, "0");    // header_option
		strcpy(argv7, "3");    // sampling-rate : 96kHz(0),48kHz(1),44.1kHz(2),32kHz(3),16kHz(4)
		strcpy(argv8, "75");    // 10ms : 100,  7.5ms : 75      // 7.5ms case
		strcpy(argv9, "31500");    // this is valid only for "header_option = 0" (no header case, argv6 == '0') 
#endif
///////////////////// 16kHz 10ms and 7.5ms and RVP mode ///////////////////////////////////////////////////
#if 0
		strcpy(argv2, "..\\..\\..\\test_vector\\km05_mus15_part_1_10.sec");
		strcpy(argv3, "..\\..\\..\\test_vector\\km05_mus15_part_1_10_dec.wav");

		strcpy(argv4, "1");    // chmode
		strcpy(argv5, "0");    //  bit-depth   0 : 16bit, 1:24bit

		strcpy(argv6, "2");    // header_option     10ms case
		strcpy(argv7, "4");    // sampling-rate : 96kHz(0),48kHz(1),44.1kHz(2),32kHz(3),16kHz(4)
		strcpy(argv8, "100");    // 10ms : 100,  7.5ms : 75     , 10ms case   
/*
		strcpy(argv6, "3");    // header_option   
		strcpy(argv7, "4");    // sampling-rate : 96kHz(0),48kHz(1),44.1kHz(2),32kHz(3),16kHz(4)
		strcpy(argv8, "75");    // 10ms : 100,  7.5ms : 75     
*/
#endif

		main_dec(argc_t, argv_t);
	}
	return EXIT_SUCCESS;
}
#endif


int main_dec(int argc, char* argv[])
{
	int err;
	char *inFile, *outFile;
	FILE *fin, *fout;

	unsigned char *out = (unsigned char *)out_pcm;
	void *dec = NULL;
	/////////////////////////////////////////////////
	int len = 0;
	int end_flag = 0;
	////////////////////////////////////////////////
	int frame_size;
	int sampling_rate;
	int tmp_bitrate;

	int skip = 0;
	int err_code = 0;
	short plc_on = 0;   //   plc_on = 0 for no PLC,    plc_on = 1 for both channel PLC.
	short ch_mode;
	int channels = 1;

	int header_option;
	int header_len;

	int bitrate_mode = 0;
	int output_bit;    // 0 : 16bit pcm out, 1 : 24bit pcm out
	WAV_HEADER		w_wav_hdr;

	inFile = argv[2];
	if (argc == 10)
	{
		if (*argv[4] == '3')
		{
			ch_mode = 3;
			channels = 2;
		}
		else if (*argv[4] == '1')
		{
			ch_mode = 1;
			channels = 1;
		}
		else
		{
			printf("strange ch_mode");
			return 0;
		}

		if (*argv[5] == '0')
		{
			output_bit = 0;
		}
		else if (*argv[5] == '1')
		{
			output_bit = 1;
		}
		else
		{
			printf("bitdepth argument should be 0(16bit) or 1(24bit)\n");
			return 0;
		}

		header_option = (int)atol(argv[6]);

		if (header_option < 0 || header_option>255)
		{
			printf("no defined header_option\n");
			return EXIT_FAILURE;
		}

		if (header_option == 0)
		{
			header_len = 0;
		}
		else if (header_option == 1)
		{
			header_len = 2;
		}
		else if (header_option == 2)
		{
			header_len = 3;
		}
		else if (header_option == 3)
		{
			header_len = 2;
		}
		else
		{
			header_len = 0;
		}

		skip = 0;
		if (*argv[7] == '0')
		{
			sampling_rate = 96000;
			frame_size = 960;
			skip = 240;
		}
		else if (*argv[7] == '1')
		{
			sampling_rate = 48000;
			frame_size = 480;
			skip = 120;
		}
		else if (*argv[7] == '2')
		{
			sampling_rate = 44100;
			frame_size = 480;
			skip = 120;
		}
		else if (*argv[7] == '3')
		{
			int t_cmp = strcmp(argv[8], "75");
			sampling_rate = 32000;
			frame_size = 320;
			skip = 80;
			if (t_cmp == 0)
			{
				frame_size = 240;
				skip = 240;
			}
		}
		else if (*argv[7] == '4')
		{
			int t_cmp = strcmp(argv[8], "75");
			sampling_rate = 16000;
			frame_size = 160;
			if (t_cmp == 0)
			{
				frame_size = 120;
			}
			skip = frame_size;
		}
		else
		{
			printf("not defined sampling rate\n");
			return 0;
		}
	}
	else
	{
		printf("check number of argument");
		return 0;
	}

	fin = fopen(inFile, "rb");
	if (!fin)
	{
		printf("can not open input file\n");
		return EXIT_FAILURE;
	}
	outFile = argv[3];
	fout = wav_open(outFile, &w_wav_hdr, WAV_WRITE);
	if (!fout)
	{
		printf("can not open output file\n");
		fclose(fin);
		return EXIT_FAILURE;
	}

/////////////////// Use inteager array til SS correct data alignment issue ///////////////////////
	dec = (void *)(dec_handle);
///////////////////////////////////////////////////////////////////////////////////////////////////////
	if (dec == NULL)
	{
		printf("can not allocate memory\n");
		fclose(fin);
		fclose(fout);
		return EXIT_FAILURE;
	}

	err = rvp_swb_decode_init(dec, (int)(ch_mode), frame_size, sampling_rate, output_bit, header_option);

	if (err < 0)
	{
		printf("can not create speech decoder handle\n");
		fclose(fin);
		fclose(fout);
		return EXIT_FAILURE;
	}

	while (1)
	{
		int output_samples;
		int tmp_len;

		int i;
		len = 0;
		for (i = 0; i < channels; i++)
		{
			err = fread(&data[len], 1, header_len, fin);
			if (err < header_len)
			{
				end_flag = 1;
			}

			if (header_option > 0)
			{
				tmp_len = rvp_swb_length_calc(&data[len], &bitrate_mode, sampling_rate, header_option, frame_size, channels);
			}
			else
			{
				tmp_bitrate = (int)atol(argv[9]);
				bitrate_mode = tmp_bitrate;
				tmp_len = tmp_bitrate * frame_size / (sampling_rate * 8);
				if (channels > 1)
				{
					tmp_len = tmp_len + (tmp_len & 1);
					tmp_len = (tmp_len >> (channels - 1));
				}
			}

			err = fread(&data[len + header_len], 1, tmp_len, fin);

			if (err < tmp_len)
			{
				end_flag = 1;
			}
			len = len + tmp_len + header_len;
		}

		if (end_flag == 1)
		{
			break;
		}
	
		plc_on = 0;

		if (plc_on == 0)
		{
			output_samples = rvp_swb_decode(dec, data, out, frame_size, plc_on, bitrate_mode, &err_code);
		}
		else
		{
			output_samples = rvp_swb_decode(dec, NULL, out, frame_size, plc_on, bitrate_mode, &err_code);
		}

		if (err_code < 0)
		{
			printf("error_code = %d\n", err_code);
		}
		if (output_samples > 0)
		{
			if (output_samples > skip)
			{
				if (output_bit == 0)
				{
					fwrite(out + (skip * 2) * channels, sizeof(short), (output_samples - skip)*channels, fout);
				}
				else
				{
					fwrite(out + (skip * 3) * channels, 3, (output_samples - skip)*channels, fout);
				}
			}
			skip = 0;
		}
		else
		{
			printf("Error is happened while decoding\n");
		}
		printf("frame decoding : %d \n", count);
		count++;
	}
	fclose(fin);
	w_wav_hdr.sample_rate = sampling_rate;// 
	w_wav_hdr.channels = channels;
	if (output_bit == 1)
	{
		w_wav_hdr.bits_per_sample = 24;
	}
	else
	{
		w_wav_hdr.bits_per_sample = 16;
	}
	wav_close(fout, &w_wav_hdr, WAV_WRITE);
	return EXIT_SUCCESS;
}

int		main_enc(int argc, char* argv[])
{
    int err;
    char *inFile, *outFile; 
    FILE *fin, *fout;
    void *enc=NULL;
    int len;
    int frame_size, channels;
    int sampling_rate;
	int bitrate;

	int input_bitdepth = 0;
    WAV_HEADER      r_wav_hdr;

	int header_option;  
//////////////////////////////////////////////////////
// 1) Sync (6bit), channel(2bit)    0xAC (Mono stream, stereo) 0xAD (Stereo/Left) right : 0xAE (Stereo/right)  0xAF (reserved)
// 2) Sampling-rate(2bit),bitrate(6bit) 
//	 : Sampling rate (2bit) : 96,48(or 44),32,16kHz sampling rate
//   : bitrate (6bit)
// 3) Sequence Number (1byte)
// 4) CRC (1byte)
// 5) high-band enenergy info (1byte)
// 6) RVP mode, sync (1byte), CRC(1byte)
///////////////////////////////////////////////////////

// header_option : 0   // No header. No 
// header option : 1   // 1) and 2)
// header option : 2   // 1) and 2) and 3)
// header option : 3   // RVP mode

//////////////////////////////////////////////////////////
	short ch_mode = 1;
//////////////////////////////////////////////////////////////////////////

	count = 0;
	if(!(  (argc == 7) || (argc == 8)) )
	{
		printf("The Number of Arguments is not correct\n");
		printf("Example : SSC_Enc_Test.exe 160 samsung.wav samsung.sec 3");
		return EXIT_FAILURE;
	}

    inFile = argv[2];
    fin = wav_open(inFile, &r_wav_hdr, WAV_READ);

    if (!fin)
    {
        fprintf (stderr, "Could not open input file %s\n", argv[2]);
        return EXIT_FAILURE;
    }

	channels = r_wav_hdr.channels;
	sampling_rate = r_wav_hdr.sample_rate;

	if(channels != 2 && channels != 1)
	{
		printf("Not supported channels\n");
		return EXIT_FAILURE;
	}

	bitrate = (int)atol(argv[5]);
	header_option = (int)atol(argv[6]);

	if (header_option < 0 || header_option>255)
	{
		printf("no defined header_option\n");
		return EXIT_FAILURE;
	}
	if (channels == 2)
	{
		ch_mode = 3;
	}

	if (r_wav_hdr.bits_per_sample == 24)
	{
		input_bitdepth = 1;
	}
	else if (r_wav_hdr.bits_per_sample == 16)
	{ 
		input_bitdepth = 0;
	}
	else
	{
		printf("Not supported bit-depth\n");
		return EXIT_FAILURE;
	}

	frame_size = 320;

	if (sampling_rate == 32000)
	{
		int t_cmp = strcmp(argv[7], "75");
		if (t_cmp == 0)
		{
			frame_size = 240;
		}
	}
	else if (sampling_rate == 16000)
	{
		int t_cmp = strcmp(argv[7], "75");
		frame_size = 160;
		if (t_cmp == 0)
		{
			frame_size = 120;
		}
	}

	outFile = argv[3];
    fout = fopen(outFile, "wb+");
    
    if (!fout)
    {
        fprintf (stderr, "Could not open output file %s\n", argv[3]);
        fclose(fin);
        return EXIT_FAILURE;
    }
///////////////////////////////////////////////////////////
//	int size = rvp_swb_encode_get_size((int)ch_mode, r_wav_hdr.sample_rate, frame_size);
	enc = (void *)(enc_handle);
///////////////////////////////////////////////////////////

	err = rvp_swb_encode_init(enc, (int)ch_mode,frame_size, r_wav_hdr.sample_rate, input_bitdepth, header_option);

	if (err < 0)
	{
		printf("rvp_swb_encode_init function is strange\n");
		return 1;
	}

    while(1)
    {		
		err = fread(in_buf, 2+input_bitdepth, frame_size*channels, fin);
		if (feof(fin))
			break;

		to_24bit_val(in, in_buf, frame_size*channels, input_bitdepth);   // packed 24bit --> extended 24bit

		len = rvp_swb_encode(enc, in, frame_size, data, bitrate);
		if (fwrite(data, 1, len, fout) != (unsigned)len) 
		{
			fprintf(stderr, "Error writing.\n");
			return EXIT_FAILURE;
		}

		printf("frame : %d\n", count);
        count++;
    }

	printf("\n finish Encoding \n");

    fclose(fin);
    fclose(fout);

    return EXIT_SUCCESS;
}
