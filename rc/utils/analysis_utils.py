import numpy as np
import os
import re
import string
from collections import Counter
import torch


def write_attns_to_file(ex, config, out_attentions, fdir, rev_word_dict):
    os.makedirs(fdir, exist_ok=True)
    for attn_type, attn in out_attentions.items():
        if attn is None:
            continue
        fp = os.path.join(fdir, '{}-{}.csv'.format(ex['id'], attn_type))
        print("Writing attention to {}".format(fp))
        _write_attn_to_file(ex, config, attn_type, attn, fp, rev_word_dict)


def _write_attn_to_file(ex, config, attn_type, attn, fp, rev_word_dict):
    attn = attn.detach().cpu().numpy()
    with open(fp, 'w') as fout:
        if attn_type != 'q_dialog_attn':
            raise NotImplementedError
        xdialog_np = ex['xdialog'].detach().cpu().numpy()
        recency_np = ex['dialog_recency_weights'].detach().cpu().numpy()
        curr_i = 0
        last_r = -1
        full_d_history = []
        if config['q_dialog_attn'] == 'word_hidden_incr':
            # Prepend keep probability.
            full_d_history.append('"<KEEP>"')
        max_r = max(recency_np[-1])
        for (last_d, r) in zip(xdialog_np[-1], recency_np[-1]):
            r = int(max_r - r)
            if r != last_r:
                last_r = r
                curr_i = 0
            full_d_history.append('"{}-{}-{}"'.format(r, curr_i, rev_word_dict[last_d]))
            curr_i += 1
        header = '"<QUESTION>",' + ','.join(full_d_history)
        fout.write(header)
        fout.write('\n')
        xq = ex['xq'].detach().cpu().numpy()
        xq_mask = ex['xq_mask'].detach().cpu().numpy()

        for q_i, (q, q_mask) in enumerate(zip(xq, xq_mask)):
            for q_token_i, (q_token, q_m) in enumerate(zip(q, q_mask)):
                if q_m.item():
                    break
                this_token = '"{}-{}-{}"'.format(q_i, q_token_i, rev_word_dict[q_token])
                attn_row = attn[q_i, q_token_i]
                this_row = '{},{}'.format(this_token, ','.join(map(str, attn_row)))
                fout.write(this_row)
                fout.write('\n')
