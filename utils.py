

def import_txt(filename):
    with open("BenchmarkScores.txt", "r") as f:
        array = []
    
        for lines in f:
            lines = lines[1:-1].split("[")
    
            for line in lines:
                bobs_exp = []
                if line:
                    line = line.split("]")
                    for l in line:
                        l = l.split(", ")
                        if l:
                            if l[0]:
                                arr = [float(i) for i in l]
                                array.append(arr)
                                
        return array