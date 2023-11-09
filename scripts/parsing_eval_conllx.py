import argparse

PUNCTS = set('!"#%&\'()*+,-.\\/:;<>?@[\]^_{|}~')


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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def eval(fp_gold, fp_sys, incl_punct=True):
    with open(fp_gold, 'r') as f_gold, open(fp_sys, 'r') as f_sys:
        no_correct_tags = 0
        no_correct_heads = 0
        no_correct_labels = 0
        total_token = 0

        f_gold = CoNLLFile(f_gold)
        f_sys = CoNLLFile(f_sys)

        for block_gold, block_sys in zip(f_gold, f_sys):
            words_gold = [parts[1] for parts in block_gold]
            words_sys = [parts[1] for parts in block_sys]
            assert words_gold == words_sys

            for parts_gold, parts_sys in zip(block_gold, block_sys):
                if incl_punct or parts_gold[1] not in PUNCTS:
                    total_token += 1
                    if parts_gold[4] == parts_sys[4]:
                        no_correct_tags += 1
                    if parts_gold[6] == parts_sys[6]:
                        no_correct_heads += 1
                        if parts_gold[7] == parts_sys[7]:
                            no_correct_labels += 1

        acc = 100 * no_correct_tags / total_token
        uas = 100. * no_correct_heads / total_token
        las = 100. * no_correct_labels / total_token

        print('ACC  = 100 * %d / %d = %5.2f' % (no_correct_tags, total_token, acc))
        print('UAS  = 100 * %d / %d = %5.2f' % (no_correct_heads, total_token, uas))
        print('LAS  = 100 * %d / %d = %5.2f' % (no_correct_labels, total_token, las))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing evaluation for CoNLL-X 2009 format')
    parser.add_argument('gold', type=str,
                        help='Gold standard file in CoNLL-X format. '
                             'Column 5 and 7 contain the gold dependency head and label.')
    parser.add_argument('sys', type=str,
                        help='Automatic parsed file in CoNLL-X format. '
                             'Column 5 and 7 contain the predicted dependency head and label.')
    parser.add_argument('-p', type=str2bool, default=True,
                        help='Evaluate with punctuations')
    args = parser.parse_args()
    eval(args.gold, args.sys, args.p)
