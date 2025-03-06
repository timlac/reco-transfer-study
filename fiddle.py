import os

l = []

for f in os.listdir('data/videos/original'):
    print(os.path.basename(f))

    l.append(os.path.splitext(f)[0])

print("[")
for item in l:
    print(f"  {repr(item)},")
print("]")