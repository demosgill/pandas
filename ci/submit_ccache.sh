#!/bin/bash

home_dir=$(pwd)
export CCACHE_DIR=$home_dir/.ccache
ccache -s

MISSES=$(ccache -s | grep "cache miss" | grep -Po "\d+")
echo "MISSES: $MISSES"

if [ x"$MISSES" == x"0" ]; then
    echo "No cache misses detected, skipping upload"
    exit 0
fi

if [ "$IRON_TOKEN" ]; then

    cd /tmp
    rm -rf ccache.7z

    tar cf - $CCACHE_DIR/.ccache \
    "$TRAVIS_BUILD_DIR"/pandas/{index,algos,lib,tslib,parser,hashtable}.c \
    "$TRAVIS_BUILD_DIR"/pandas/src/{sparse,testing}.c \
    "$TRAVIS_BUILD_DIR"/pandas/msgpack.cpp  \
    |  7za a -si ccache.7z

    split -b 500000 -d ccache.7z ccache.

    python ci/ironcache/put.py
    cd $home_dir
fi;

exit 0
