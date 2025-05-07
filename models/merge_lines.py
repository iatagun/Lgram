import re

input_path  = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\more.txt'   # buraya kendi dosyanın adını yaz
output_path = 'output.txt'

with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

merged = []
buffer = ''

for line in lines:
    text = line.rstrip('\n')
    # Eğer sonu nokta, soru işareti, ünlem, tırnak veya apostrof ile bitiyorsa cümle sonu varsay
    if re.search(r'[\.!?]["’’]?\s*$', text):
        buffer = (buffer + ' ' + text).strip()
        merged.append(buffer)
        buffer = ''
    else:
        # cümle devam ediyor: tampona ekle
        buffer = (buffer + ' ' + text).strip()

# Eğer sona kalmış bir tampon varsa ekleyelim
if buffer:
    merged.append(buffer)

# Yazdır
with open(output_path, 'w', encoding='utf-8') as f:
    for line in merged:
        f.write(line + '\n')
