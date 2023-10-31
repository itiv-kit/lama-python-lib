from lama.checkpoints import lama_compare_checkpoint

minmax = [[20,30],[20,30]]

lama_compare_checkpoint(minmax, 't1/checkpoint1.pkl', is_online=True)