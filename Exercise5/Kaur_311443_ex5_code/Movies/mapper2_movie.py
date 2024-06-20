#!/usr/bin/python39
import sys
import pandas as pd
input = sys.stdin
next(input)                  # ignore the column names
for line in input:
    user_id = line.split(';')[3]             # user
    rating = line.split(';')[4]                    # rating
    if rating:                       # if raitng is not None
        print(user_id, ';', rating)