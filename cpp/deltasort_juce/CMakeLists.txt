# SPDX-License-Identifier: MIT
#
# Benchmark target that links real JUCE (juce_audio_basics module) and uses
# the actual juce::MidiMessageSequence / juce::MidiMessage classes.
#
# Build:
#   mkdir build && cd build
#   cmake -DCMAKE_BUILD_TYPE=Release -DJUCE_DIR=/path/to/JUCE ..
#   cmake --build .
#   ./juce_midi_bench

cmake_minimum_required(VERSION 3.22)

project(deltasort_juce_bench VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

if(NOT DEFINED JUCE_DIR)
    set(JUCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/JUCE"
        CACHE PATH "Path to a JUCE source checkout")
endif()

if(NOT EXISTS "${JUCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "JUCE_DIR='${JUCE_DIR}' does not contain a CMakeLists.txt. "
        "Pass -DJUCE_DIR=/path/to/JUCE pointing at a JUCE checkout.")
endif()

# Pull JUCE in as a subdirectory build.
add_subdirectory("${JUCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/JUCE_build" EXCLUDE_FROM_ALL)

juce_add_console_app(juce_midi_bench
    PRODUCT_NAME "DeltaSort JUCE MIDI Benchmark")

target_sources(juce_midi_bench PRIVATE juce_midi_bench.cpp)

target_compile_definitions(juce_midi_bench PRIVATE
    JUCE_USE_CURL=0
    JUCE_WEB_BROWSER=0
    JUCE_STANDALONE_APPLICATION=0)

target_link_libraries(juce_midi_bench PRIVATE
    juce::juce_audio_basics
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags)

target_include_directories(juce_midi_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")
