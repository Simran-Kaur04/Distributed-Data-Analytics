#!/usr/bin/python39
import sys
input = sys.stdin
next(input)                          # ignore the column names
for line in input:
    movie_title = line.split(';')[1]
    rating = line.split(';')[4]
    # if rating is not None then print it
    if rating:
        print(movie_title, ';', rating)