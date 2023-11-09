def write_tuba_dz(fp, raw_data, results):
    with open(fp, 'w') as f:
        for i, result in enumerate(results):
            f.write('%s %s %s %s %s %s %s ' % (raw_data['sentence_nums'][i],
                                               raw_data['prep_words'][i],
                                               raw_data['prep_tags'][i],
                                               raw_data['prep_tfs'][i],
                                               raw_data['obj_words'][i],
                                               raw_data['obj_tags'][i],
                                               raw_data['obj_tfs'][i]))
            no_candidates = len(raw_data['can_flags'][i])
            prediction = [0] * no_candidates
            prediction[result] = 1
            s = list()
            for j in range(no_candidates):
                s.append('%s %s %s %d %d %d' % (raw_data['can_words'][i][j],
                                                raw_data['can_tags'][i][j],
                                                raw_data['can_tfs'][i][j],
                                                raw_data['can_abs_dists'][i][j],
                                                raw_data['can_rel_dists'][i][j],
                                                prediction[j]))
            f.write(' '.join(s))
            f.write('\n')
