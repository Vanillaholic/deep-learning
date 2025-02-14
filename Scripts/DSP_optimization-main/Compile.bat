setlocal

::Directories

set Work_Dir="D:\Projects\RVP_SWB-main"
set Inc_Dir="D:\Projects\RVP_SWB-main\source_code\inc"
set Src_Dir="D:\Projects\RVP_SWB-main\source_code\src"
set Build_Dir="D:\Projects\RVP_SWB-main\Build"
set Lib_Dir="D:\Projects\RVP_SWB-main\lib"

::test文件没有设置
set Test_Code_Dir="D:\Projects\RVP_SWB-main\test_main"


::set Tool_Dir_Gnu=C:\Qualcomm\HEXAGON_Tools\8.0.13\Tools\bin
set Tool_Dir_Gnu=C:\Qualcomm\HEXAGON_Tools\8.0.13\Tools\bin
::set Tool_Dir_qc="C:\Qualcomm\HEXAGON_Tools\8.0.13\Tools\bin"
set Tool_Dir_qc=C:\Qualcomm\HEXAGON_Tools\8.0.13\Tools\bin

::compiler options
set CFLAGS=-mv62 -O2 -Wall -Werror -Wno-cast-align -Wpointer-arith -Wno-missing-braces -Wno-strict-aliasing -fno-exceptions -fno-strict-aliasing -fno-zero-initialized-in-bss -fdata-sections  -Wstrict-prototypes -Wnested-externs -DCORTEX_M3_OPT -DKHW_CODE_OPT -DSTACK_OPT
set CFLAGS_SRC=-mv62 -v -O2 -Wall -Werror -Wno-cast-align -Wpointer-arith -Wno-missing-braces -Wno-strict-aliasing -fno-exceptions -fno-strict-aliasing -fno-zero-initialized-in-bss -fdata-sections -mllvm -disable-hsdr -Wstrict-prototypes -Wnested-externs -DRVDS_ON -DCORTEX_M3_OPT -DKHW_CODE_OPT -DSTACK_OPT
set CFLAGS_MAIN=-mv62 -v -O2 -Wall -Werror -Wno-cast-align -Wpointer-arith -Wno-missing-braces -Wno-strict-aliasing -fno-exceptions -fno-strict-aliasing -fno-zero-initialized-in-bss -fdata-sections -mllvm -disable-hsdr -Wstrict-prototypes -Wnested-externs -DRVDS_ON -DBLE_AUDIO_ONLY -DINSERT_HEADER -DALIGIN_BUFFER
CLS

cd %Work_Dir%


ECHO *************Building started****************
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_audio_decoder.o %Src_Dir%\rvp_swb_audio_decoder.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_audio_encoder.o %Src_Dir%\rvp_swb_audio_encoder.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_bands_dec.o %Src_Dir%\rvp_swb_bands_dec.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_bands_enc.o %Src_Dir%\rvp_swb_bands_enc.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_cwrs.o %Src_Dir%\rvp_swb_cwrs.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_fft.o %Src_Dir%\rvp_swb_fft.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_laplace.o %Src_Dir%\rvp_swb_laplace.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_lpc.o %Src_Dir%\rvp_swb_lpc.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_mathops.o %Src_Dir%\rvp_swb_mathops.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_mdct.o %Src_Dir%\rvp_swb_mdct.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_modes.o %Src_Dir%\rvp_swb_modes.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_pitch_util.o %Src_Dir%\rvp_swb_pitch_util.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_pitch.o %Src_Dir%\rvp_swb_pitch.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_quant.o %Src_Dir%\rvp_swb_quant.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_rate.o %Src_Dir%\rvp_swb_rate.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_rc_code.o %Src_Dir%\rvp_swb_rc_code.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_rc_decode.o %Src_Dir%\rvp_swb_rc_decode.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_rc_encode.o %Src_Dir%\rvp_swb_rc_encode.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_ssc.o %Src_Dir%\rvp_swb_ssc.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_util.o %Src_Dir%\rvp_swb_util.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_SRC% -I%Inc_Dir% -c -o %Build_Dir%\rvp_swb_vector_quant.o %Src_Dir%\rvp_swb_vector_quant.c


%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_MAIN% -I%Test_Code_Dir% -c -o %Test_Code_Dir%\audio_wav.o %Test_Code_Dir%\audio_wav.c
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS_MAIN% -I%Test_Code_Dir% -I%Inc_Dir% -c -o %Test_Code_Dir%\rvp_swb_main_release.o %Test_Code_Dir%\rvp_swb_main_release.c

ECHO ***************Objects Created*******************
PAUSE
ECHO ***************start generate library***************** **
%Tool_Dir_Gnu%\hexagon-ar -rsc %Lib_Dir%\rvp_swb.lib  %Build_Dir%\rvp_swb_audio_decoder.o %Build_Dir%\rvp_swb_audio_encoder.o %Build_Dir%\rvp_swb_bands_dec.o %Build_Dir%\rvp_swb_bands_enc.o %Build_Dir%\rvp_swb_cwrs.o %Build_Dir%\rvp_swb_fft.o %Build_Dir%\rvp_swb_laplace.o %Build_Dir%\rvp_swb_lpc.o %Build_Dir%\rvp_swb_mathops.o %Build_Dir%\rvp_swb_mdct.o %Build_Dir%\rvp_swb_modes.o %Build_Dir%\rvp_swb_pitch_util.o %Build_Dir%\rvp_swb_pitch.o %Build_Dir%\rvp_swb_quant.o %Build_Dir%\rvp_swb_rate.o %Build_Dir%\rvp_swb_rc_code.o %Build_Dir%\rvp_swb_rc_decode.o %Build_Dir%\rvp_swb_rc_encode.o %Build_Dir%\rvp_swb_ssc.o %Build_Dir%\rvp_swb_util.o %Build_Dir%\rvp_swb_vector_quant.o 
ECHO *************End generate library*********************
PAUSE
ECHO *************Linking started*********************

%Tool_Dir_Gnu%\hexagon-size -d %Lib_Dir%\rvp_swb.lib  > result_data_size_totals.txt
%Tool_Dir_Gnu%\hexagon-clang %CFLAGS% %Test_Code_Dir%\audio_wav.o %Test_Code_Dir%\rvp_swb_main_release.o %Lib_Dir%\rvp_swb.lib  -o %Build_Dir%\RVP_SWB -lhexagon

ECHO *********************Compilation Done*************************
PAUSE


endlocal