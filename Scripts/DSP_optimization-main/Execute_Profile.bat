setlocal

::Directories
::路径
set Work_Dir="D:\Projects\RVP_SWB-main"
set Build_Dir="D:\Projects\RVP_SWB-main\Build"


::set Tool_Dir_Gnu=C:\Qualcomm\HEXAGON_Tools\8.0.13\Tools\bin
set Tool_Dir_Gnu=C:\Qualcomm\HEXAGON_Tools\8.0.13\Tools\bin
::set Tool_Dir_qc="C:\Qualcomm\HEXAGON_Tools\8.0.13\Tools\bin"  
set Tool_Dir_qc=C:\Qualcomm\HEXAGON_Tools\8.0.13\Tools\bin


::FLAGS
::set CFLAGS=-O3 -D__unix -DALIGN_GNU_ARM -mv55


CLS
D:
cd %Work_Dir%
PAUSE
ECHO OFF
::Run Simulator
ECHO **************************Simulation started************************
ECHO ON
%Tool_Dir_qc%\hexagon-sim.exe %Build_Dir%\RVP_SWB --profile
ECHO OFF
ECHO **************************Simulation finished***********************
ECHO ON
PAUSE
::Run Profiler
ECHO OFF
ECHO ***************************Profiling started*************************
ECHO ON
::%Tool_Dir_Gnu%\qdsp6-gprof.exe %Build_Dir%\AoBLE_QDSP gmon.t_0>gprofile2.txt
%Tool_Dir_qc%\hexagon-gprof.exe --profile %Build_Dir%\RVP_SWB gmon.t_0>gprofile2.txt
::%Tool_Dir_qc%\qdsp6-cov.exe -i %Build_Dir%\SCALABLE_QDSP gmon.t_0 -o coverage2.txt
ECHO OFF
ECHO ***************************Profiling finished************************
ECHO ON

PAUSE

endlocal