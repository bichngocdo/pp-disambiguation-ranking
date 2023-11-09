import argparse

PUNCTS = '-!"#%&\'()*,.\\/:;?@[\\\]_{}'


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


def eval(fp, incl_punct=True):
    with open(fp, 'r') as f:
        no_correct_tags = 0
        no_correct_heads = 0
        no_correct_labels = 0
        total_token = 0

        f = CoNLLFile(f)
        for block in f:
            for parts in block:
                if incl_punct or parts[1] not in PUNCTS:
                    total_token += 1
                    if parts[3] == parts[4]:
                        no_correct_tags += 1
                    if parts[6] == parts[8]:
                        no_correct_heads += 1
                        if parts[7] == parts[9]:
                            no_correct_labels += 1

        acc = 100 * no_correct_tags / total_token
        uas = 100. * no_correct_heads / total_token
        las = 100. * no_correct_labels / total_token

        print('ACC  = 100 * %d / %d = %5.2f' % (no_correct_tags, total_token, acc))
        print('UAS  = 100 * %d / %d = %5.2f' % (no_correct_heads, total_token, uas))
        print('LAS  = 100 * %d / %d = %5.2f' % (no_correct_labels, total_token, las))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing evaluation for CoNLL 2009 format')
    parser.add_argument('input', type=str,
                        help='Input file in CoNLL 2009 format. '
                             'Column 9 and 10 contain the gold and predicted dependency heads, '
                             'column 11 and 12 contain the gold and predicted dependency labels.')
    parser.add_argument('-p', type=str2bool, default=True,
                        help='Evaluate with punctuations')
    args = parser.parse_args()
    eval(args.input, args.p)
