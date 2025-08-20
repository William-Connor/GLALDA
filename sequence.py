import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import time
import http

# Set your email here
Entrez.email = "huzhenkang8733033@gmail.com"
Entrez.api_key = "b528be83fd4db1cf3f37b5a7aeee368c2c09"

# Load the data from the .xlsx file
df = pd.read_excel('data/my data/01-lncRNAs-240.xlsx', header=None)
#df = pd.read_excel('./01-lncRNAs-240.xlsx', header=None)
# Get the list of lncRNA names
lncrnas = df.iloc[:, 0].tolist()

# Dictionary to store lncRNA download status
lncrna_status = {lncrna: 'NA' for lncrna in lncrnas}

# List to store SeqRecord objects
records = []

# For each lncRNA, download the sequence
for lncrna in lncrnas:
    for i in range(10):  # retry up to 10 times
        try:
            handle = Entrez.esearch(db="nucleotide", term=f"{lncrna}[Gene] AND Homo sapiens[Organism]", retmax=1)
            record = Entrez.read(handle)
            handle.close()

            if int(record["Count"]) > 0:
                handle = Entrez.efetch(db="nucleotide", id=record["IdList"][0], rettype="fasta", retmode="text")
                seq_record = SeqIO.read(handle, "fasta")
                handle.close()

                # Create a SeqRecord and add it to the list
                records.append(SeqRecord(Seq(str(seq_record.seq)), id=lncrna, description=""))

                # Update the status of lncRNA
                lncrna_status[lncrna] = 'Downloaded'
                print(f"Sequence for {lncrna} has been downloaded.")
            else:
                print(f"No sequence found for {lncrna}")
                # Add a 'NA' SeqRecord for lncRNA not found
                records.append(SeqRecord(Seq('NA'), id=lncrna, description=""))
        except http.client.IncompleteRead:
            print(f"Network error occurred while fetching {lncrna}, retrying...")
            time.sleep(2)  # wait for 2 seconds before retrying
            continue
        break  # if no error occurred, break the loop and proceed with the next lncrna

# Write the SeqRecord objects to a FASTA file
SeqIO.write(records, "lncrna_sequences.fa", "fasta")

# Write the lncRNA status to a .txt file
with open('lncrna_status.txt', 'w') as f:
    for lncrna, status in lncrna_status.items():
        f.write(f"{lncrna}\t{status}\n")

# import pandas as pd
# import requests
# from bs4 import BeautifulSoup
# import time
#
# # Load the data from the .xlsx file
# df = pd.read_excel('data/my data/01-lncRNAs-240.xlsx',header=None)
#
# # Get the list of lncRNA names
# lncrnas = df.iloc[:, 0].tolist()
#
# # List to store NCBI IDs
# ncbi_ids = []
#
# # For each lncRNA, search for its NCBI ID
# for lncrna in lncrnas:
#     time.sleep(1)  # to prevent overwhelming the server with requests
#     response = requests.get(f"https://www.google.com/search?q={lncrna}+NCBI+id")
#     soup = BeautifulSoup(response.text, 'html.parser')
#     # Extract the NCBI ID from the search results
#     for result in soup.find_all('div', class_='BNeawe'):
#         if 'Gene ID:' in result.text:
#             ncbi_id = result.text.split('Gene ID:')[1].split(',')[0]
#             ncbi_ids.append(ncbi_id)
#             break
#     else:
#         print(f"No NCBI ID found for {lncrna}")
#         ncbi_ids.append(None)
#
# # Create a DataFrame with the lncRNA names and their NCBI IDs
# df_ncbi_ids = pd.DataFrame({'lncRNA': lncrnas, 'NCBI ID': ncbi_ids})
#
# # Save the DataFrame to a .txt file
# df_ncbi_ids.to_csv('ncbi_ids.txt', index=False)
