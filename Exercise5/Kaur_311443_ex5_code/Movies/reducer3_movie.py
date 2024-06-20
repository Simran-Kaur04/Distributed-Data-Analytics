#!/usr/bin/python39
import sys
input = sys.stdin
current_genre = None   # to check if current and next genre are same
current_rating = 0              # to calculate rating for each genre
avg_rating = {}                    # average rating per genre
count = 0
for line in input:
    genre = line.split(';')[0]
    rating = line.split(';')[1]
    try:
        rating = float(rating)
    except ValueError:
        continue
    if genre == current_genre:
        current_rating += rating
        count += 1
    else:
        if current_genre:
            avg_rating[current_genre] = current_rating/count
            count = 0
        else:
            print('Highest average rated movie genre:')
        current_genre = genre
        current_rating = rating
        count += 1
if current_genre == genre:
    avg_rating[current_genre] = current_rating/count

genre_rating = [k for k, v in sorted(avg_rating.items(), key=lambda item: item[1], reverse= True)][0]
print(genre_rating)