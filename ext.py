import numpy as np

def multi_lan_cleanup(trg_words, zw, lan_tracker, remove_lan_from_target):
    if len(remove_lan_from_target) == 0:
        return (trg_words, zw)
    if len(lan_tracker) < len(remove_lan_from_target):
        raise Exception("why are you removing more than what I have?")
    for lan in remove_lan_from_target:
        try:
            start_idx, end_idx = lan_tracker[lan]
            trg_words[:] = [trg_word for i, trg_word in enumerate(trg_words) if not (start_idx <= i < end_idx)]
            zw           = np.delete(zw, np.s_[start_idx:end_idx], axis=0)
        except KeyError:
            raise Exception("why do you want me to remove what I don't have?")
    return (trg_words, zw)
