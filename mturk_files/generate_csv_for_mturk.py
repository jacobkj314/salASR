#import pandas as pd

audios_per_hit = 18

where_digits = ['t','b','r']
r_digits = ['2','5','8']

def audio(hit, a):
    """
    Returns the name of the (a)'th audio file for the 
    (hit)'th hit
    """
    block = hit // 9
    hit = hit % 9


    instance = audios_per_hit*block + a

    where = where_digits[(a + (hit // 3)) % 3]

    r = r_digits[(a + hit) % 3]

    return f'https://jacob.ml/salASR/audios/{instance}_{r}_{where}.wav'

with open("mturk.csv", "w") as writer:
    writer.write(",".join(f"audio{i}" for i in range(audios_per_hit))+"\n")
    for hit in range(18):
        writer.write(",".join(audio(hit, a) for a in range(audios_per_hit))+"\n")


