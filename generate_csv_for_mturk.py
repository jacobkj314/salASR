#import pandas as pd

instance_digits = [0,1,2,3,4,5]
where_digits = ['t','b','r']
r_digits = ['2','5','8']

def audio(hit, a):
    """
    Returns the name of the (a)'th audio file for the 
    (hit)'th hit
    """
    block = hit // 9
    hit = hit % 9


    instance = 6*block + a

    where = where_digits[(a + (hit % 3)) % 3]

    r = r_digits[(a + hit) % 3]

    return f'https://jacob.ml/salASR/audios/{instance}_{where}_{r}.wav'

with open("mturk.csv", "w") as writer:
    writer.write(",".join(f"audio{i}" for i in range(6))+"\n")
    for hit in range(90):
        writer.write(",".join(audio(hit, a) for a in range(6))+"\n")


