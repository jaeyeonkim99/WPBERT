import sys

if __name__=="__main__":

    model = sys.argv[1]
    metric = sys.argv[2]

    f0 = open(f'/data/babymind/{model}/{metric}/test0.txt', 'r')
    f1 = open(f'/data/babymind/{model}/{metric}/test1.txt', 'r')
    f2 = open(f'/data/babymind/{model}/{metric}/test2.txt', 'r')
    f3 = open(f'/data/babymind/{model}/{metric}/test3.txt', 'r')

    write = open(f'/data/babymind/{model}/{metric}/test.txt', 'w')


while True:
    line0 = f0.readline()
    line1 = f1.readline()
    line2 = f2.readline()
    line3 = f3.readline()

    if not line0: break

    write.write(line0)
    if not not line1: write.write(line1)
    if not not line2: write.write(line2)
    if not not line3: write.write(line3)

f0.close()
f1.close()
f2.close()
f3.close()
write.close()