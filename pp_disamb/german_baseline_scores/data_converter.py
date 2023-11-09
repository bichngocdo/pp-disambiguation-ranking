from pp_disamb.data.data_converters import AbstractDataConverter, convert_to_tensor


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
        results.append(convert_to_tensor(batch[10]))
        results.append(convert_to_tensor(batch[11]))
        results.append(convert_to_tensor(batch[12]))
        results.append(convert_to_tensor(batch[13]))
        results.append(convert_to_tensor(batch[14]))
        results.append(convert_to_tensor(batch[15]))
        results.append(convert_to_tensor(batch[16]))
        results.append(convert_to_tensor(batch[17], type='float32'))
        # Labels
        results.append(convert_to_tensor(batch[18], type='float32', value=-1))
        return results
