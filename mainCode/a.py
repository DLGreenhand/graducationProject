for i in range(10):
    with open('a.txt','a') as fp:
        fp.write(f"{i}\n")