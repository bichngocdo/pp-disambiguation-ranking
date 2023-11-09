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
                items = line.split('\t')
                if not line and self.sentence:
                    sentences = self.sentence
                    self.sentence = []
                    return sentences
                if line:
                    self.sentence.append(items)

    def close(self):
        self.file.close()


def eval(fp_gold, fp_pred, no_NAs=0, only_nv=True):
    no_correct = 0
    no_incorrect = 0
    no_gold = 0

    with open(fp_gold, 'r') as f_gold, open(fp_pred, 'r') as f_pred:
        f_gold = CoNLLFile(f_gold)

        line = f_pred.readline()

        for block in f_gold:
            for parts in block:
                if parts[3] in {'APPR', 'APPRART', 'APPO'} \
                        and (not only_nv or block[int(parts[6]) - 1][3][0] in {'N', 'V'}):
                    no_gold += 1

            if line is None:
                break

            gold_sentence_id = int(block[0][0].split('_')[0])

            while line != '':
                pred_parts = line.rstrip().split(' ')
                pred_sentence_id, pp = pred_parts[0].split('_')
                pred_sentence_id = int(pred_sentence_id)
                pp = int(pp)
                if pred_sentence_id == gold_sentence_id:
                    pred_head = -1
                    for i in range(7, len(pred_parts), 6):
                        if int(pred_parts[i + 5]) == 1:
                            pred_head = pp + int(pred_parts[i + 3])
                            break
                    gold_head = int(block[pp - 1][6])
                    if block[pp - 1][3] in {'APPR', 'APPRART', 'APPO'} \
                            and (not only_nv or block[int(block[pp - 1][6]) - 1][3][0] in {'N', 'V'}):
                        if gold_head == pred_head:
                            no_correct += 1
                        else:
                            no_incorrect += 1
                elif pred_sentence_id > gold_sentence_id:
                    break
                line = f_pred.readline()

    no_pred = no_correct + no_incorrect

    prec = 100. * no_correct / no_pred
    rec = 100. * no_correct / no_gold
    f1 = 2 * prec * rec / (prec + rec)

    print('NA-disc:')
    print('   P: 100. * %d / %d = %5.2f' % (no_correct, no_pred, prec))
    print('   R: 100. * %d / %d = %5.2f' % (no_correct, no_gold, rec))
    print('  F1: %5.2f' % f1)
    print()

    no_pred += no_NAs
    prec = 100. * no_correct / no_pred
    f1 = 2 * prec * rec / (prec + rec)

    print('NA-error:')
    print('   P: 100. * %d / %d = %5.2f' % (no_correct, no_pred, prec))
    print('   R: 100. * %d / %d = %5.2f' % (no_correct, no_gold, rec))
    print('  F1: %5.2f' % f1)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PP disambiguation evaluation')
    parser.add_argument('gold', type=str,
                        help='The gold file in CoNLL format. '
                             'Column 3 contains the POS tag used to identify prepositions, '
                             'and column 7 contains the dependency heads. '
                             'Column 1 contains token ids in format {sentence_id}_{token_id}.')
    parser.add_argument('pred', type=str,
                        help='The predicted file in format defined in de Kok et al., 2017')
    parser.add_argument('--na', type=int, default=0,
                        help='Number of non-attachment cases needs to be taken into account')
    parser.add_argument('--only_nv', action='store_true',
                        help='Evaluate only on prepositions that have nouns or verbs as objects')
    args = parser.parse_args()

    eval(args.gold, args.pred, args.na, args.only_nv)
