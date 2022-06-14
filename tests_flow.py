import dwt
import lbph
import json

weights = []
r_rate = []


class Results:
    def __init__(self, weights, result):
        self.weights = weights
        self.result = result


for i in range(1, 10000):
    with open('results/data.json', 'a') as outfile:
        w = dwt.create_samples()
        weights.append(w)
        lbph.train()
        result = lbph.lbph_verification_tests()
        r_rate.append(result)

        data = {'weights': w, "result": result}
        json.dump(data, outfile)
        outfile.write('\n')
        outfile.close()
