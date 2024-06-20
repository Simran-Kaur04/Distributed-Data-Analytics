#!/usr/bin/python39
import sys
import pandas as pd
input = sys.stdin
next(input)
for line in input:       
    l_genre = []        # if one movie has more than one genre
    genres = line.split(';')[2]
    rating = line.split(';')[4]
    try:
        genres = genres.strip()
        l_genre = genres.split('|')    # split the genres
        for g in l_genre:
            if rating:
                print(g, ';', rating)    # give each of the genre same rating as they belong to one movie
    except:
        if rating:
            print(genres, ';', rating)
