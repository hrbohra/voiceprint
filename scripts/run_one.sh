#!/usr/bin/env bash
# scripts/run_one.sh <logname> <args...>: one eval, detached-friendly, logs to scratch/<logname>.log
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
name=$1; shift
echo "START $(date +%T)" > scratch/$name.log
.venv/Scripts/python.exe "$@" >> scratch/$name.log 2>&1
echo "EXIT $? $(date +%T)" >> scratch/$name.log
