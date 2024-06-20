import sys
input = sys.stdin
current_movie = None        # this variable is to check whether current and next movie are same
current_rating = 0                     # to calculate ratings for each movie
avg_rating = {}                   # dictionary to store average ratings for each movie
count = 0
for line in input:
    movie = line.split(';')[0]
    rating = line.split(';')[1]
    try:
        rating = float(rating)                     # if the rating is available then only convert to float
    except ValueError:
        continue
    if movie == current_movie:
        current_rating += rating
        count += 1
    else:
        if current_movie:
            avg_rating[current_movie] = current_rating/count
            count = 0
        else:
            print('Movie with highest average rating:')
        current_movie = movie
        current_rating = rating
        count += 1
if current_movie == movie:
    avg_rating[current_movie] = current_rating/count

top_movie = max(avg_rating.items(), key=lambda item: item[1])     # movie with the max average rating
for key, value in avg_rating.items():
    if value == top_movie[1]:
        print(key, value)