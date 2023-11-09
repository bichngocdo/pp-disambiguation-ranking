import numpy as np

from pp_disamb.data.data_converters import AbstractDataConverter


def convert_to_tensor(sequences, type='int32'):
    return np.asarray(sequences).astype(type)


def convert_to_tensor_padded(sequences, type='int32', value=0):
    lengths = [len(s) for s in sequences]
    batch_size = len(sequences)
    max_length = max(lengths)
    tensor = np.zeros((batch_size, max_length)).astype(type) + value
    for i, s in enumerate(sequences):
        tensor[i, :lengths[i]] = s
    return tensor


class DataConverter(AbstractDataConverter):
    def __init__(self):
        super(AbstractDataConverter, self).__init__()

    def convert(self, batch):
        results = list()
        # Prepositions
        results.append(convert_to_tensor(batch[0]).reshape(-1, 1))
        results.append(convert_to_tensor(batch[1]).reshape(-1, 1))
        results.append(convert_to_tensor(batch[2]).reshape(-1, 1))
        results.append(convert_to_tensor(batch[3]).reshape(-1, 1))
        results.append(convert_to_tensor(batch[4]).reshape(-1, 1))
        # Objects
        results.append(convert_to_tensor(batch[5]).reshape(-1, 1))
        results.append(convert_to_tensor(batch[6]).reshape(-1, 1))
        results.append(convert_to_tensor(batch[7]).reshape(-1, 1))
        results.append(convert_to_tensor(batch[8]).reshape(-1, 1))
        results.append(convert_to_tensor(batch[9]).reshape(-1, 1))
        # Candidates
        results.append(convert_to_tensor_padded(batch[10]))
        results.append(convert_to_tensor_padded(batch[11]))
        results.append(convert_to_tensor_padded(batch[12]))
        results.append(convert_to_tensor_padded(batch[13]))
        results.append(convert_to_tensor_padded(batch[14]))
        results.append(convert_to_tensor_padded(batch[15]))
        results.append(convert_to_tensor_padded(batch[16]))
        # Labels
        results.append(convert_to_tensor_padded(batch[17], type='float32', value=-1))
        return results
