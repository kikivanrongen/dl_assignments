file = 'hamlet.txt'
hamlet100 = open('hamlet100.txt', 'w')
for i in range(n):
    with open(file, 'r') as hamlet:
        lines = hamlet.readlines()
        text = ''.join(lines)
        hamlet100.write(text)
hamlet100.close()
