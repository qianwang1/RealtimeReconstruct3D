# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/troll/Programs/clion-2017.2.1/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/troll/Programs/clion-2017.2.1/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/troll/workspace/RealtimeReconstruct3D

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/troll/workspace/RealtimeReconstruct3D/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/RealtimeReconstruct3D.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RealtimeReconstruct3D.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RealtimeReconstruct3D.dir/flags.make

CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o: CMakeFiles/RealtimeReconstruct3D.dir/flags.make
CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/troll/workspace/RealtimeReconstruct3D/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o -c /home/troll/workspace/RealtimeReconstruct3D/src/main.cpp

CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/troll/workspace/RealtimeReconstruct3D/src/main.cpp > CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.i

CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/troll/workspace/RealtimeReconstruct3D/src/main.cpp -o CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.s

CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o.requires

CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o.provides: CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/RealtimeReconstruct3D.dir/build.make CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o.provides

CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o.provides.build: CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o


# Object files for target RealtimeReconstruct3D
RealtimeReconstruct3D_OBJECTS = \
"CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o"

# External object files for target RealtimeReconstruct3D
RealtimeReconstruct3D_EXTERNAL_OBJECTS =

RealtimeReconstruct3D: CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o
RealtimeReconstruct3D: CMakeFiles/RealtimeReconstruct3D.dir/build.make
RealtimeReconstruct3D: /opt/pylon5/lib64/libbxapi-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libgxapi-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylonbase-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylonc-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_bcon-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_camemu-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_gige-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_gtc-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_usb-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylonutility-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libuxapi-5.0.9.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libbxapi.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libFirmwareUpdate_gcc_v3_0_Basler_pylon_v5_0.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libGCBase_gcc_v3_0_Basler_pylon_v5_0.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libGenApi_gcc_v3_0_Basler_pylon_v5_0.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libgxapi.so
RealtimeReconstruct3D: /opt/pylon5/lib64/liblog4cpp_gcc_v3_0_Basler_pylon_v5_0.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libLog_gcc_v3_0_Basler_pylon_v5_0.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libMathParser_gcc_v3_0_Basler_pylon_v5_0.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libNodeMapData_gcc_v3_0_Basler_pylon_v5_0.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylonbase.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylonc.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_bcon.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_camemu.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_gige.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_gtc.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylon_TL_usb.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libpylonutility.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libuxapi.so
RealtimeReconstruct3D: /opt/pylon5/lib64/libXmlParser_gcc_v3_0_Basler_pylon_v5_0.so
RealtimeReconstruct3D: /opt/pylon5/lib64/pylon-libusb-1.0.so
RealtimeReconstruct3D: /usr/local/lib/libopencv_dnn.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_ml.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_objdetect.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_shape.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_stitching.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_superres.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_videostab.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_calib3d.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_features2d.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_flann.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_highgui.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_photo.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_video.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_videoio.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_imgproc.so.3.3.0
RealtimeReconstruct3D: /usr/local/lib/libopencv_core.so.3.3.0
RealtimeReconstruct3D: CMakeFiles/RealtimeReconstruct3D.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/troll/workspace/RealtimeReconstruct3D/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable RealtimeReconstruct3D"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RealtimeReconstruct3D.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RealtimeReconstruct3D.dir/build: RealtimeReconstruct3D

.PHONY : CMakeFiles/RealtimeReconstruct3D.dir/build

CMakeFiles/RealtimeReconstruct3D.dir/requires: CMakeFiles/RealtimeReconstruct3D.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/RealtimeReconstruct3D.dir/requires

CMakeFiles/RealtimeReconstruct3D.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RealtimeReconstruct3D.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RealtimeReconstruct3D.dir/clean

CMakeFiles/RealtimeReconstruct3D.dir/depend:
	cd /home/troll/workspace/RealtimeReconstruct3D/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/troll/workspace/RealtimeReconstruct3D /home/troll/workspace/RealtimeReconstruct3D /home/troll/workspace/RealtimeReconstruct3D/cmake-build-debug /home/troll/workspace/RealtimeReconstruct3D/cmake-build-debug /home/troll/workspace/RealtimeReconstruct3D/cmake-build-debug/CMakeFiles/RealtimeReconstruct3D.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RealtimeReconstruct3D.dir/depend

