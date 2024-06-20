#!/usr/bin/python39
import sys
input = sys.stdin
current_airport = None   # this variable is to check whether current and next item are same
current_delay = 0                   # to store the total delay for each airport
count = 0
airports = {}        # this dictionary will store airport as keys and average delay as value
top_airport = []           # list containing 10 top airports
for line in input:
    line = line.strip()
    airport = line.split(',')[0]
    delay = line.split(',')[1]
    try:
        delay = float(delay)
    except ValueError:
        continue
    if delay > 0:
        if current_airport == airport:
            current_delay += delay
            count += 1
        else:
            if current_airport:
                airports[current_airport] = current_delay/count
                count = 0
            else:
                print('Top 10 Airports')
            current_airport = airport
            current_delay = delay
            count += 1

if current_airport == airport:
    airports[current_airport] = current_delay/count
top_airport = [(k, v) for k, v in sorted(airports.items(), key=lambda item: item[1])][:10]
for tup in top_airport:
    print(tup)