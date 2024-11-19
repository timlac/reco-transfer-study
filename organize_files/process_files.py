from glob import glob
from pathlib import Path
import re

from nexa_sentimotion_filename_parser.metadata import Metadata


def clean_filename(filename):
    return re.sub(r'_V1.*', '', filename)

# kanske 2 stycken från A72 och en från A426

# for p in glob("data/*.mov"):
#     print(p)

for p in Path("../data").glob("*.mov"):


    print(p)
    print(p.stem)
    print(clean_filename(p.stem))

    # m = Metadata(p.stem)

    # print(vars(m))

