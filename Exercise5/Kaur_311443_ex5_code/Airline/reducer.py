#!/usr/bin/python39
import sys
input = sys.stdin
current_airport = None      # this variable is to check whether current and next item are same
current_delay = 0                 # to store the total delay for each airport
count = 0
for line in input:
    line = line.strip()
    airport = line.split(',')[0]
    delay = line.split(',')[1]
    try:
        delay = float(delay)
    except ValueError:
        continue
    # this is to do calculation for the same airport
    if delay > 0:
        if current_airport == airport:
            current_delay += delay
            count += 1
            if min_delay > delay:
                min_delay = delay
            if max_delay < delay:
                max_delay = delay
        else:
            if current_airport:        # now when the airport is different we need to print everything for the previous airport
                print(current_airport, ',', max_delay, ',', min_delay, ',', current_delay/count)
                count = 0
            else:
                print('Airport, Max Delay, Min Delay, Average Delay')
            current_airport = airport
            current_delay = delay
            count += 1
            min_delay = delay
            max_delay = delay
# for the last line
if current_airport == airport:
    print(current_airport, ',', max_delay, ',', min_delay, ',', current_delay/count)

    