#!/usr/bin/python39
import sys
input = sys.stdin
current_user = None      # this variable is to check whether current and next user are same
current_rating = 0                     # to calculate ratings assigned by each user
avg_rating = {}                     # average rating per user
count = 0
for line in input:
    user = line.split(';')[0]
    rating = line.split(';')[1]
    try:
        rating = float(rating)
    except ValueError:
        continue
    if user == current_user:
        current_rating += rating
        count += 1
    else:
        if (current_user is not None) and (count > 40):           # if user is not None and has rated more than 40 movies
            avg_rating[current_user] = current_rating/count
            count = 0
        elif current_user is None:
            print('Users who has assigned lowest average rating:')
        current_user = user
        current_rating = rating
        count += 1
if current_user == user and count > 40:
    avg_rating[current_user] = current_rating/count

lowest_rating = min(avg_rating.items(), key=lambda item: item[1])     # users with lowest average rating
for key, value in avg_rating.items():
    if value == lowest_rating[1]:
        print(key, value)
