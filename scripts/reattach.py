import argparse


class CoNLLFile:
    def __init__(self, f):
        self.file = f
        self.sentence = list()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            line = self.file.readline()
            if not line:
                if not self.sentence:
                    raise StopIteration
                else:
                    sentences = self.sentence
                    self.sentence = []
                    return sentences
            else:
                line = line.rstrip()
                if not line and self.sentence:
                    sentences = self.sentence
                    self.sentence = []
                    return sentences
                if line:
                    self.sentence.append(line)

    def close(self):
        self.file.close()


def _read_results(f):
    results = list()
    for line in f:
        parts = line.split(' ')
        sentence_id, pp_id = parts[0].split('_')
        sentence_id = int(sentence_id)
        pp_id = int(pp_id)
        for i in range(7, len(parts), 6):
            if parts[i + 5] == '1':
                head_id = int(parts[i + 3]) + pp_id
                results.append((sentence_id, pp_id, head_id))
                break
    return results


def is_noun_or_verb(tag):
    return tag[0] in {'N', 'V'}


PPOS_COL = 4
PHEAD_COL = 8


def reattach(fp_gold, fp_pred, fp_out, only_nv=False):
    with open(fp_gold, 'r') as f_gold, open(fp_pred, 'r') as f_pred, open(fp_out, 'w') as f_out:
        results = _read_results(f_pred)
        k = 0
        f_gold = CoNLLFile(f_gold)
        for block in f_gold:
            block = [line.split('\t') for line in block]
            sentence_id = int(block[0][0].split('_')[0])
            while k < len(results) and sentence_id > results[k][0]:
                k += 1
            while k < len(results) and sentence_id == results[k][0]:
                reattach_sentence_id, reattach_pp_id, reattach_head_id = results[k]
                current_head_id = int(block[reattach_pp_id - 1][PHEAD_COL])
                current_head_tag = block[current_head_id - 1][PPOS_COL]
                if not only_nv or is_noun_or_verb(current_head_tag):
                    block[reattach_pp_id - 1][PHEAD_COL] = str(reattach_head_id)
                k += 1
            f_out.write('\n'.join(['\t'.join(parts) for parts in block]))
            f_out.write('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PP reattachment')
    parser.add_argument('gold', type=str,
                        help='The output of a dependency parser in CoNLL format. '
                             'Column 9 contains the predicted dependency heads.')
    parser.add_argument('pred', type=str,
                        help='The predicted file in format defined in de Kok et al., 2017')
    parser.add_argument('output', type=str,
                        help='The reattachment result file')
    parser.add_argument('--only_nv', action='store_true',
                        help='Reattach only prepositions of which the current head are nouns or verbs')
    args = parser.parse_args()

    reattach(args.gold, args.pred, args.output, only_nv=args.only_nv)
