__all__ = ['read_tuba_dz']


def read_preposition(parts):
    prep_word = parts[1]
    prep_tag = parts[2]
    prep_tf = parts[3]
    return prep_word, prep_tag, prep_tf


def read_object(parts):
    obj_word = parts[4]
    obj_tag = parts[5]
    obj_tf = parts[6]
    return obj_word, obj_tag, obj_tf


def read_candidates(parts):
    can_words = list()
    can_tags = list()
    can_tfs = list()
    can_abs_dists = list()
    can_rel_dists = list()
    can_flags = list()
    for i in range(7, len(parts), 6):
        can_words.append(parts[i])
        can_tags.append(parts[i + 1])
        can_tfs.append(parts[i + 2])
        can_abs_dists.append(int(parts[i + 3]))
        can_rel_dists.append(int(parts[i + 4]))
        can_flags.append(int(parts[i + 5]))
    return can_words, can_tags, can_tfs, can_abs_dists, can_rel_dists, can_flags


def read_tuba_dz(fp):
    with open(fp, 'r') as f:
        all_sentence_nums = list()
        all_prep_words = list()
        all_prep_tags = list()
        all_prep_tfs = list()
        all_obj_words = list()
        all_obj_tags = list()
        all_obj_tfs = list()
        all_can_words = list()
        all_can_tags = list()
        all_can_tfs = list()
        all_can_abs_dists = list()
        all_can_rel_dists = list()
        all_can_flags = list()

        for line in f:
            parts = line.rstrip().split(' ')
            prep_word, prep_tag, prep_tf = read_preposition(parts)
            obj_word, obj_tag, obj_tf = read_object(parts)
            can_words, can_tags, can_tfs, can_abs_dists, can_rel_dists, can_flags = read_candidates(parts)
            all_sentence_nums.append(parts[0])
            all_prep_words.append(prep_word)
            all_prep_tags.append(prep_tag)
            all_prep_tfs.append(prep_tf)
            all_obj_words.append(obj_word)
            all_obj_tags.append(obj_tag)
            all_obj_tfs.append(obj_tf)
            all_can_words.append(can_words)
            all_can_tags.append(can_tags)
            all_can_tfs.append(can_tfs)
            all_can_abs_dists.append(can_abs_dists)
            all_can_rel_dists.append(can_rel_dists)
            all_can_flags.append(can_flags)

        return {
            'sentence_nums': all_sentence_nums,
            'prep_words': all_prep_words,
            'prep_tags': all_prep_tags,
            'prep_tfs': all_prep_tfs,
            'obj_words': all_obj_words,
            'obj_tags': all_obj_tags,
            'obj_tfs': all_obj_tfs,
            'can_words': all_can_words,
            'can_tags': all_can_tags,
            'can_tfs': all_can_tfs,
            'can_abs_dists': all_can_abs_dists,
            'can_rel_dists': all_can_rel_dists,
            'can_flags': all_can_flags
        }
