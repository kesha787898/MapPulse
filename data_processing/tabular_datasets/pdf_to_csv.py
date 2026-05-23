import pdfplumber
import pandas as pd

from paths import LIVEABILITY_CSV, LIVEABILITY_PDF

file = LIVEABILITY_PDF

all_tables = []

with pdfplumber.open(file) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            df = pd.DataFrame(table[1:], columns=table[0])
            all_tables.append(df)
all_tables = all_tables[1:]
for table in all_tables:
    columns = [str(i[0] if i[0] else "") + "" + str(i[1] if i[1] else "") for i in zip(table.iloc[0], table.iloc[1])]
    table.columns = columns
    table.drop([0, 1], axis=0, inplace=True)
final = pd.concat(all_tables)
final.to_csv(LIVEABILITY_CSV, index=False)
