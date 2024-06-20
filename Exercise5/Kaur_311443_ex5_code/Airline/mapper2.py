#!/usr/bin/python39
import sys
# Storing the file in input
input = sys.stdin
for line in input:
    line = line.strip()
    airport = line.split(',')[3]
    delay = line.split(',')[6]
    # if delay is not None then print it
    if delay:
        print(airport, ',', delay)